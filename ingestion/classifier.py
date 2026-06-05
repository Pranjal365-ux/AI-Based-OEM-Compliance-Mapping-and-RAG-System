# =============================================================================
# classifier.py — Robust category detection for OEM documents
# =============================================================================

import re
import logging
CATEGORY_TAXONOMY = {
    "NGFW": {
        "keywords": [
            "next-generation firewall", "next generation firewall", "ngfw",
            "fortigate", "palo alto networks", "pa-series", "strata", "fortios",
            "stateful inspection", "app-id", "user-id", "wildfire",
            "threat prevention", "ips throughput", "firewall throughput",
            "ssl inspection", "zero trust network", "ztna",
            "security fabric", "fortimanager", "panorama",
            "ml-powered firewall", "fortiguard", "firewall policy",
            "next-gen firewall", "network firewall",
        ],
        "title_words": [
            "fortigate", "palo alto", "ngfw", "fortios",
            "ml-powered", "next-generation firewall", "network firewall",
        ],
        "negative": ["web application firewall", "waf", "load balancer", "adc"],
    },
    "WAF": {
        "keywords": [
            "web application firewall", "waf", "owasp top 10",
            "sql injection", "cross-site scripting", "xss",
            "appwall", "f5 advanced waf", "big-ip asm",
            "application security manager", "bot defense", "bot mitigation",
            "credential stuffing", "layer 7 attack", "csrf",
            "behavioral dos", "advanced waf", "silverline",
            "radware appwall", "virtual patching", "web scraping prevention",
        ],
        "title_words": [
            "appwall", "advanced waf", "web application firewall", "waf",
            "big-ip asm",
        ],
        "negative": ["next-generation firewall", "ngfw", "load balancer"],
    },
    "ADC": {
        "keywords": [
            "application delivery controller", "adc", "load balancer",
            "load balancing", "alteon", "big-ip ltm", "local traffic manager",
            "ssl offload", "server load balancing", "global server load balancing",
            "gslb", "tcp multiplexing", "http pooling", "content switching",
            "viprion", "radware alteon", "citrix adc",
            "layer 4", "layer 7 load", "virtual server",
        ],
        "title_words": [
            "alteon", "ltm", "load balancer", "adc", "application delivery",
        ],
        "negative": ["web application firewall", "waf", "ngfw"],
    },
    "IPS": {
        "keywords": [
            "intrusion prevention", "intrusion detection", "ips", "ids",
            "signature-based detection", "anomaly detection", "snort",
            "suricata", "network-based ips", "nips", "inline ips",
            "zero-day intrusion", "exploit detection",
            "defenseflow", "tipping point",
        ],
        "title_words": [
            "ips", "intrusion prevention", "ids", "defenseflow",
        ],
        "negative": ["firewall", "waf", "load balancer"],
    },
    "DDoS": {
        "keywords": [
            "ddos", "denial of service", "syn flood", "udp flood",
            "volumetric attack", "scrubbing", "arbor",
            "netscout", "anti-ddos", "ddos mitigation",
            "flood protection", "attack mitigation",
        ],
        "title_words": [
            "ddos", "defenseflow", "scrubbing", "flood protection",
        ],
        "negative": [],
    },
    "SWITCH": {
        "keywords": [
            "network switch", "ethernet switch", "layer 2", "layer 3 switch",
            "vlan", "spanning tree", "stp", "rstp", "lacp", "lldp",
            "catalyst", "nexus", "aruba", "juniper ex",
            "fortiswitch", "802.1q", "port channel",
        ],
        "title_words": [
            "switch", "catalyst", "nexus", "fortiswitch",
        ],
        "negative": ["firewall", "router", "waf"],
    },
    "ROUTER": {
        "keywords": [
            "router", "routing", "bgp", "ospf", "mpls", "sd-wan",
            "wan edge", "asr", "juniper mx", "vrf", "route reflector",
            "fortiwan", "silverpeak", "velocloud",
        ],
        "title_words": [
            "router", "sd-wan", "wan", "asr", "mx series",
        ],
        "negative": ["switch", "firewall"],
    },
    "ENDPOINT": {
        "keywords": [
            "endpoint security", "edr", "endpoint detection",
            "antivirus", "anti-malware", "crowdstrike", "symantec",
            "trend micro", "mcafee", "forticlient", "host-based",
            "endpoint protection", "agent-based",
        ],
        "title_words": [
            "endpoint", "edr", "crowdstrike", "forticlient",
        ],
        "negative": ["network", "firewall"],
    },
    "APT": {
        "keywords": [
            "advanced persistent threat", "apt",
            "sandbox", "malware analysis", "zero-day",
            "threat intelligence", "advanced malware protection",
            "deep discovery", "sandboxing"
        ],
        "title_words": [
            "apt", "sandbox", "deep discovery"
        ],
        "negative": []
    },

    "ZTNA": {
        "keywords": [
            "ztna", "zero trust network access",
            "secure remote access", "identity-based access",
            "least privilege access"
        ],
        "title_words": [
            "ztna", "zero trust"
        ],
        "negative": []
    },

    "DLP": {
        "keywords": [
            "data loss prevention", "dlp",
            "data leakage prevention",
            "sensitive data protection",
            "information protection"
        ],
        "title_words": [
            "dlp", "data loss prevention"
        ],
        "negative": []
    },

    "NAC": {
        "keywords": [
            "network access control", "nac",
            "device profiling",
            "guest access",
            "endpoint posture",
            "device visibility"
        ],
        "title_words": [
            "nac"
        ],
        "negative": []
    },

    "WIRELESS_AP": {
        "keywords": [
            "wireless access point",
            "access point",
            "wifi",
            "wifi 6",
            "wifi 6e",
            "wifi 7",
            "802.11ax",
            "802.11ac",
            "wireless controller"
        ],
        "title_words": [
            "access point",
            "wireless",
            "wifi"
        ],
        "negative": []
    },

    "OSINT_DARKWEB": {
        "keywords": [
            "dark web",
            "deep web",
            "osint",
            "open source intelligence",
            "threat intelligence feed",
            "threat actor monitoring",
            "credential leak monitoring"
        ],
        "title_words": [
            "osint",
            "dark web"
        ],
        "negative": []
    },

    "STORAGE": {
        "keywords": [
            "storage",
            "san",
            "nas",
            "object storage",
            "block storage",
            "storage array"
        ],
        "title_words": [
            "storage"
        ],
        "negative": []
    },

    "HCI": {
        "keywords": [
            "hyperconverged",
            "hyper-converged",
            "hci",
            "virtualized infrastructure"
        ],
        "title_words": [
            "hci",
            "hyperconverged"
        ],
        "negative": []
    },

    "SERVER": {
        "keywords": [
            "server",
            "blade server",
            "rack server",
            "compute node",
            "poweredge",
            "proliant"
        ],
        "title_words": [
            "server"
        ],
        "negative": []
    },

    "CLOUD_SERVICES": {
        "keywords": [
            "cloud service",
            "iaas",
            "paas",
            "saas",
            "cloud-native",
            "cloud infrastructure"
        ],
        "title_words": [
            "cloud"
        ],
        "negative": []
    },

    "CSPM_CWPP": {
        "keywords": [
            "cspm",
            "cwpp",
            "cloud posture",
            "cloud workload protection",
            "container security",
            "runtime protection"
        ],
        "title_words": [
            "cspm",
            "cwpp"
        ],
        "negative": []
    },

    "OT_SECURITY": {
        "keywords": [
            "ot security",
            "industrial security",
            "ics",
            "scada",
            "operational technology"
        ],
        "title_words": [
            "ot",
            "scada"
        ],
        "negative": []
    },

    "PIM_PAM": {
        "keywords": [
            "pam",
            "pim",
            "privileged access management",
            "privileged identity management",
            "credential vault"
        ],
        "title_words": [
            "pam",
            "pim"
        ],
        "negative": []
    },

    "SSL_CERTIFICATE": {
        "keywords": [
            "ssl certificate",
            "tls certificate",
            "certificate lifecycle",
            "certificate authority"
        ],
        "title_words": [
            "certificate"
        ],
        "negative": []
    },

    "KEY_MANAGEMENT_HSM": {
        "keywords": [
            "hsm",
            "hardware security module",
            "key management",
            "cryptographic key"
        ],
        "title_words": [
            "hsm"
        ],
        "negative": []
    },

    "LOG_MANAGEMENT": {
        "keywords": [
            "log management",
            "syslog",
            "event logging",
            "log analytics"
        ],
        "title_words": [
            "log"
        ],
        "negative": [
            "siem"
        ]
    },

    "APPLICATION_PERFORMANCE_MONITORING_SEARCH": {
        "keywords": [
            "application performance monitoring",
            "apm",
            "observability",
            "distributed tracing",
            "elasticsearch",
            "search platform"
        ],
        "title_words": [
            "apm",
            "observability"
        ],
        "negative": []
    },

    "SIEM": {
        "keywords": [
            "siem",
            "security information and event management",
            "security analytics",
            "event correlation",
            "ueba",
            "soc",
            "threat hunting"
        ],
        "title_words": [
            "siem"
        ],
        "negative": []
    }


}


logger = logging.getLogger("ingestion")


def _normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _score_category(cat_name: str, cat_def: dict,
                    filename_norm: str, head_norm: str, body_norm: str) -> float:
    score = 0.0
    for kw in cat_def.get("keywords", []):
        kw_n = _normalise(kw)
        if kw_n in filename_norm: score += 3.0
        if kw_n in head_norm:     score += 2.0
        if kw_n in body_norm:     score += 1.0
    for tw in cat_def.get("title_words", []):
        if _normalise(tw) in filename_norm:
            score += 5.0
    for neg in cat_def.get("negative", []):
        neg_n = _normalise(neg)
        if neg_n in filename_norm: score -= 4.0
        if neg_n in head_norm:     score -= 2.0
    return max(score, 0.0)


def detect_category(filename: str, full_text: str):
    filename_norm = _normalise(filename)
    head_norm     = _normalise(full_text[:1000])
    body_norm     = _normalise(full_text)

    scores = {
        cat: _score_category(cat, defn, filename_norm, head_norm, body_norm)
        for cat, defn in CATEGORY_TAXONOMY.items()
    }
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_cat, top_score   = ranked[0]
    second_score         = ranked[1][1] if len(ranked) > 1 else 0.0

    if top_score == 0.0:
        logger.warning(f"  [classifier] No match for '{filename}' → GENERAL")
        return "GENERAL", 0.0

    total      = top_score + second_score
    confidence = (top_score - second_score) / total if total > 0 else 0.0

    if confidence < 0.35:
        logger.warning(
            f"  [classifier] Low confidence ({confidence:.2f}) '{filename}' "
            f"→ {top_cat} ({top_score:.1f}) vs {ranked[1][0]} ({second_score:.1f})"
        )

    logger.info(f"  [classifier] '{filename}' -> {top_cat} (score={top_score:.1f}, conf={confidence:.2f})")
    return top_cat, round(confidence, 3)


def propagate_category_from_models(
    doc_category: str,
    doc_confidence: float,
    model_categories: list,
) -> tuple:
    """
    If the document-level category detection failed (Unknown / zero confidence),
    fall back to the majority category from the per-model entries.

    Parameters
    ----------
    doc_category    : category returned by detect_category for the full document
    doc_confidence  : confidence score returned by detect_category
    model_categories: list of (category, confidence) tuples from each ModelSpec

    Returns
    -------
    (resolved_category, resolved_confidence)
    """
    if doc_category != "Unknown" and doc_confidence > 0.0:
        return doc_category, doc_confidence

    if not model_categories:
        return doc_category, doc_confidence

    # Tally votes weighted by per-model confidence
    tally: dict = {}
    for cat, conf in model_categories:
        if cat and cat != "Unknown":
            tally[cat] = tally.get(cat, 0.0) + (conf or 0.5)

    if not tally:
        return doc_category, doc_confidence

    best_cat = max(tally, key=lambda c: tally[c])
    # Average confidence across models that voted for the winner
    winning_confs = [conf for cat, conf in model_categories if cat == best_cat]
    avg_conf = sum(winning_confs) / len(winning_confs) if winning_confs else 0.5

    logger.info(
        f"  [classifier] document category resolved via model propagation: "
        f"{best_cat} (avg_conf={avg_conf:.2f})"
    )
    return best_cat, round(avg_conf, 3)