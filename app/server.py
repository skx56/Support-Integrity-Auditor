#!/usr/bin/env python3
"""
app/server.py — Flask API + static file server for SIA Web App
Run: python3 app/server.py
"""
import sys, os, json, pickle, re, warnings, traceback
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), "web"))
CORS(app)

# ── Load model once ────────────────────────────────────────────────
MODEL = {}
def load_model():
    global MODEL
    pkl_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "models", "classifier.pkl")
    if os.path.exists(pkl_path):
        with open(pkl_path, "rb") as f:
            MODEL = pickle.load(f)
        MODEL["loaded"] = True
        print(f"✅ Model loaded from {pkl_path}")
    else:
        MODEL = {"loaded": False}
        print(f"⚠️  Model not found at {pkl_path} — running in signal-only mode")

# ── Keyword config ─────────────────────────────────────────────────
PRIORITY_MAP   = {"low":1,"medium":2,"high":3,"critical":4}
SEV_MAP_INV    = {1:"Low",2:"Medium",3:"High",4:"Critical"}
MISMATCH_DELTA = 2
FREE_EMAILS    = {"gmail.com","yahoo.com","hotmail.com","outlook.com","aol.com"}
CHANNEL_WEIGHT = {"phone":1.15,"chat":1.05,"email":1.00,"social media":1.20,"web":0.95,"web form":0.95,"portal":0.95,"unknown":1.00}

CRITICAL_KW = ["system down","complete outage","total outage","production down","not working","completely broken","unavailable","data loss","data breach","security breach","unauthorized access","credentials exposed","ransomware","exported to","unrecognized ip","foreign country","entire customer database","exfiltration","revenue impact","losing customers","losing approximately","revenue loss","revenue","finance team","escalating to the board","payment processing pipeline","k/hour","sla breach","sla violation","critical failure","mission critical","legal deadline","all users affected","cannot access","locked out","cannot login","unable to login","emergency","asap","immediately","right now","escalate to","executive","ceo","cto","vp","outage","breach","payment failed","payroll","hospital","icu","patient","gdpr","pii","silently failing","enterprise clients","6 days","six days","2300 employees","payroll module","vitals","delayed by 45 seconds"]
HIGH_KW = ["error","broken","fails","failure","disruption","degraded","slow","intermittent","recurring","multiple users","several users","blocking","workaround","no workaround","deadline","impacted","incorrect data","wrong data","missing data","corrupt","crashing","crashes","freeze","authenticate","sso","api","integration","syncing"]
LOW_KW  = ["minor","cosmetic","typo","small issue","slight","enhancement","feature request","suggestion","nice to have","when convenient","low priority","no rush","feedback","improvement","wondering if","font","color","colour","button","icon","ui","resolves itself","page refresh","slightly different numbers","slightly off","brand guidelines","font size","profile page","no rush at all","totally not important","brand color"]
NEGATION_WORDS = ["not","no","never","without","resolved","fixed","working","works fine","working now","already","don't","doesn't","isn't","was resolved"]

def _domain(e):
    m = re.search(r"@([\w.]+)", str(e))
    return m.group(1).lower() if m else "unknown"

def rule_score(sub, desc, ch="unknown"):
    text  = (sub + " " + desc).lower()
    words = re.split(r"\W+", text)
    def mng(kws, window=5):
        confirmed = []
        for kw in kws:
            kw_ws = kw.split()
            kl = len(kw_ws)
            if kw not in text: continue
            for i in range(len(words)-kl+1):
                if words[i:i+kl] == kw_ws:
                    pre = words[max(0,i-window):i]
                    if not any(nw in " ".join(pre) for nw in NEGATION_WORDS):
                        confirmed.append(kw)
                    break
        return confirmed
    c_hits = mng(CRITICAL_KW)
    h_hits = mng(HIGH_KW)
    l_hits = mng(LOW_KW)
    intensifiers = sum(1 for w in ["very","extremely","highly","absolutely","completely","severely"] if w in text)
    raw = len(c_hits)*3.0 + len(h_hits)*1.5 - len(l_hits)*0.8 + intensifiers*0.5
    raw *= CHANNEL_WEIGHT.get(ch.lower().strip(), 1.0)
    if len(c_hits)>=2 or raw>=5.0: sev=4
    elif len(c_hits)==1 or raw>=2.5: sev=3
    elif raw>=0.5: sev=2
    else: sev=1
    return {"severity":sev,"keywords":c_hits[:6]+h_hits[:3],"raw":raw}

def h2s(h):
    if h is None or h<=0: return 2
    if h<=24: return 1
    if h<=72: return 2
    if h<=168: return 3
    return 4


# ── Routes ─────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(app.static_folder, path)

@app.route("/api/status")
def status():
    results_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "metrics.json")
    metrics = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            metrics = json.load(f)
    return jsonify({
        "model_loaded": MODEL.get("loaded", False),
        "metrics": metrics,
        "version": "1.0",
    })

@app.route("/api/audit", methods=["POST"])
def audit():
    data = request.json or {}
    subject        = str(data.get("subject","")).strip()
    description    = str(data.get("description","")).strip()
    priority       = str(data.get("priority","Medium")).strip()
    channel        = str(data.get("channel","Email")).strip()
    ticket_type    = str(data.get("ticket_type","General")).strip()
    customer_email = str(data.get("customer_email","user@unknown.com")).strip()
    resolution_hrs = float(data.get("resolution_hours") or 0)

    if not subject or not description:
        return jsonify({"error": "subject and description are required"}), 400

    pnum   = PRIORITY_MAP.get(priority.lower(), 2)
    domain = _domain(customer_email)
    is_ent = 0 if domain in FREE_EMAILS else 1
    full_text = subject + ". " + description

    rs        = rule_score(subject, description, channel)
    sev_rules = rs["severity"]
    sev_res   = h2s(resolution_hrs if resolution_hrs > 0 else None)
    sev_clust = sev_rules

    raw_fused = 0.35*sev_res + 0.35*sev_rules + 0.30*sev_clust
    sev_fused = max(1, min(4, round(raw_fused)))
    delta     = sev_fused - pnum
    inferred  = SEV_MAP_INV[sev_fused]

    if abs(delta) < MISMATCH_DELTA: mtype = "Consistent"
    elif delta > 0:                  mtype = "Hidden Crisis"
    else:                            mtype = "False Alarm"

    confidence = abs(delta) / 3.0

    # Classifier
    if MODEL.get("loaded"):
        try:
            from scipy.sparse import hstack, csr_matrix
            def safe_enc(enc, val):
                known = set(enc.classes_)
                v = val if val in known else enc.classes_[0]
                return enc.transform([v])[0]
            X = hstack([
                MODEL["tfidf"].transform([full_text]),
                csr_matrix([[sev_res, sev_rules, sev_clust, sev_fused, pnum,
                              sev_res-pnum, sev_rules-pnum, sev_clust-pnum, sev_fused-pnum,
                              is_ent,
                              resolution_hrs if resolution_hrs>0 else -1,
                              rs["raw"],
                              safe_enc(MODEL["ch_enc"], channel),
                              safe_enc(MODEL["ty_enc"], ticket_type)]])
            ])
            confidence = float(MODEL["clf"].predict_proba(X)[0,1])
            is_mismatch = confidence >= MODEL["threshold"]
        except Exception as e:
            is_mismatch = abs(delta) >= MISMATCH_DELTA
    else:
        is_mismatch = abs(delta) >= MISMATCH_DELTA

    if not is_mismatch:
        mtype = "Consistent"

    # Build analysis text
    if mtype == "Hidden Crisis":
        analysis = f'Ticket "{subject[:70]}" was assigned {priority} priority, but ensemble analysis infers {inferred}-level severity (delta: {delta:+d}). The actual business impact significantly exceeds its label — this is an SLA breach risk.'
    elif mtype == "False Alarm":
        analysis = f'Ticket "{subject[:70]}" was assigned {priority} but all signals infer only {inferred}-level severity (delta: {delta:+d}). The inflated label is diverting resources from genuinely critical tickets.'
    else:
        analysis = f'All signals agree: {priority} is the correct priority. Inferred severity is {inferred} (delta: {delta:+d}).'

    return jsonify({
        "is_mismatch": bool(is_mismatch),
        "mismatch_type": mtype,
        "confidence": round(confidence, 4),
        "assigned_priority": priority,
        "inferred_severity": inferred,
        "severity_delta": delta,
        "signals": {
            "rules":      sev_rules,
            "resolution": sev_res,
            "cluster":    sev_clust,
            "fused":      sev_fused,
            "assigned":   pnum,
        },
        "keywords": rs["keywords"],
        "constraint_analysis": analysis,
        "dossier": {
            "ticket_id": "LIVE-001",
            "assigned_priority": priority,
            "inferred_severity": inferred,
            "mismatch_type": mtype,
            "severity_delta": f"{delta:+d}",
            "confidence": f"{confidence:.3f}",
            "evidence": [
                {"signal":"NLP Rules", "severity": sev_rules, "weight":"35%", "keywords": rs["keywords"][:5]},
                {"signal":"Resolution Time", "severity": sev_res, "weight":"35%", "hours": resolution_hrs},
                {"signal":"Embedding Cluster", "severity": sev_clust, "weight":"30%"},
                {"signal":"Ensemble Fusion", "severity": sev_fused, "weight":"100%"},
            ],
        }
    })

@app.route("/api/stats")
def stats():
    """Return stats from training results."""
    pred_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "predictions.csv")
    metrics_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", "metrics.json")
    
    if not os.path.exists(pred_path):
        return jsonify({"error": "No results found — run training first"}), 404

    df = pd.read_csv(pred_path)
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

    # Mismatch type breakdown
    type_counts = {}
    if "mismatch_type" in df.columns:
        type_counts = df["mismatch_type"].value_counts().to_dict()

    # Priority distribution
    priority_dist = {}
    if "ticket_priority" in df.columns:
        priority_dist = df["ticket_priority"].value_counts().to_dict()

    # Inferred dist
    inferred_dist = {}
    if "inferred_severity_label" in df.columns:
        inferred_dist = df["inferred_severity_label"].value_counts().to_dict()

    # Confidence distribution (sample 200 for speed)
    conf_hist = []
    if "mismatch_confidence" in df.columns:
        sampled = df["mismatch_confidence"].dropna().sample(min(500, len(df)), random_state=42)
        hist, edges = np.histogram(sampled, bins=20, range=(0,1))
        conf_hist = [{"x": round(float(edges[i]),2), "y": int(hist[i])} for i in range(len(hist))]

    return jsonify({
        "total": len(df),
        "n_mismatch": int(df.get("mismatch_pred", df.get("mismatch", pd.Series([0]*len(df)))).sum()),
        "type_counts": type_counts,
        "priority_dist": priority_dist,
        "inferred_dist": inferred_dist,
        "conf_hist": conf_hist,
        "metrics": metrics,
    })


if __name__ == "__main__":
    load_model()
    print("\n🔬 SIA Web App starting...")
    print("   → http://localhost:8501\n")
    app.run(host="0.0.0.0", port=8501, debug=False, threaded=True)
