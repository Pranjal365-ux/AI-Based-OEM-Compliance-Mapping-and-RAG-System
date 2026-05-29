# =============================================================================
# config.py — Central configuration for the RAG ingestion pipeline
# =============================================================================

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_PATH = os.path.join(BASE_DIR, "oem_docs")
DB_PATH   = os.path.join(BASE_DIR, "vectordb")
LOG_PATH  = os.path.join(BASE_DIR, "logs", "ingestion.log")

# ── Embedding ─────────────────────────────────────────────────────────────────
# BGE-base gives significantly better retrieval quality than MiniLM for
# technical/enterprise content. Runs fully locally via sentence-transformers.
# Swap to "BAAI/bge-small-en-v1.5" if memory is a constraint.
EMBED_MODEL = "BAAI/bge-base-en-v1.5"
COLLECTION  = "oem_knowledge_base"

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE        = 512
CHUNK_OVERLAP     = 64
MIN_CHUNK_CHARS   = 80
DEDUP_THRESHOLD   = 0.92   # Jaccard similarity above which a chunk is a duplicate

# ── OCR ───────────────────────────────────────────────────────────────────────
OCR_CHAR_THRESHOLD = 50    # chars below which a page is treated as image-based
OCR_DPI            = 200

# ── Category taxonomy ─────────────────────────────────────────────────────────
# Each entry:
#   keywords    — scored across filename (×3), first-1000-chars (×2), full body (×1)
#   title_words — strong bonus (+5) when found in filename
#   negative    — penalty when found in filename (−4) or opening section (−2)

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
