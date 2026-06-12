# taxonomy.py
"""
Product Category Taxonomy
=========================
Used by TaxonomyClassifier in rfp_extractor.py.

Scoring rules (per category per chunk):
  title_words match (first 200 chars of chunk)  →  +3 pts each
  keywords match (anywhere in chunk)             →  +1 pt each
  negative match (anywhere in chunk)             →  -5 pts each

Assign to highest-scoring category with score >= MIN_SCORE (default 2).
If no category reaches MIN_SCORE → LLM fallback.

Keywords have been enriched from the Integrated Data Center RFP
(datacenter_2023-12-29) in addition to the original vendor-product terms.
"""

CATEGORY_TAXONOMY = {

    # ── Network Security ──────────────────────────────────────────────────────

    "NGFW": {
        "keywords": [
            # original vendor / product terms
            "next-generation firewall", "next generation firewall", "ngfw",
            "fortigate", "palo alto networks", "pa-series", "strata", "fortios",
            "stateful inspection", "app-id", "user-id", "wildfire",
            "threat prevention", "ips throughput", "firewall throughput",
            "ssl inspection", "zero trust network", "ztna",
            "security fabric", "fortimanager", "panorama",
            "ml-powered firewall", "fortiguard", "firewall policy",
            "next-gen firewall", "network firewall",
            # from RFP text
            "dedicated purpose built firewall", "convergence of high performing networking",
            "ai/ ml-powered services", "application visibility and control",
            "web security", "content security", "malware control",
            "active/active", "active/passive", "ipv4 & ipv6", "high-availability",
            "12x 1ge rj45", "8 x 1ge sfp", "8x 10g sfp+",
            "mix / production traffic", "threat prevention throughput",
            "site-to-site vpn", "client to site vpn",
            "ssl vpn users", "new sessions per second", "concurrent sessions",
            "nat64", "dns64", "dhcpv6", "virtual domains", "virtual firewall domains",
            "transparent mode", "nat/route mode", "virtual contexts",
            "traffic shaping", "ipsec vpn", "pptp vpn", "l2tp vpn",
            "hardware vpn acceleration", "multi-zone vpn", "hub and spoke",
            "redundant gateway", "route based ipsec", "policy based ipsec",
            "ipv6 ipsec", "virtual domain", "antivirus solution",
            "av signatures", "file blocking", "web content filtering",
            "web url block", "keyword block", "110 million", "rated websites",
            "4000 application signatures", "application control list",
            "active/standby", "ether channel", "redundant power supply",
            "240 gb", "hard drive capacity", "two factor authentication",
            "full feature parity", "centralized management tool",
            "integrated redundant power",
        ],
        "title_words": [
            "fortigate", "palo alto", "ngfw", "fortios",
            "ml-powered", "next-generation firewall", "network firewall",
            "next generation firewall",
        ],
        "negative": ["web application firewall", "waf", "load balancer", "adc",
                     "intrusion prevention system", "ips appliance"],
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
            # original
            "application delivery controller", "adc", "load balancer",
            "load balancing", "alteon", "big-ip ltm", "local traffic manager",
            "ssl offload", "server load balancing", "global server load balancing",
            "gslb", "tcp multiplexing", "http pooling", "content switching",
            "viprion", "radware alteon", "citrix adc",
            "layer 4", "layer 7 load", "virtual server",
            # from RFP text
            "local application switching", "tcp optimization",
            "filter-based load balancing", "transparent deployments",
            "content-based load balancing", "persistency",
            "http content modifications", "global server load balancing",
            "tls versions and cipher controls",
            "http/2", "http/3", "gateway",
            "ospf", "rip1", "rip2", "bgp",        # routing on ADC
            "vrrp", "rfc - 2338",                  # HA on ADC
            "minimum misses", "hash", "persistent hash", "tunable hash",
            "weighted hash", "least connections", "round-robin", "response time",
            "virtual matrix architecture",
            "client network address translation", "proxy ip",
            "mapping ports", "direct server return", "one arm topology",
            "direct access mode", "multiple ip addresses",
            "immediate and delayed binding",
            "user-written scripts", "application flows",
            "dnssec", "global load balancing",
            "proximity based llb", "full-path transaction",
            "application-aware full-path health monitoring",
            "latency, packet loss",
            "static nat", "dynamic nat", "no-nat",
            "layer 7 load balancing", "layer 7 content switch",
            "caching", "client side rtt", "server side rtt",
            "traffic rendering time",
            "2 x 10 ge", "8 x 1 ge rj45",
            "5 gbps", "25 gbps", "500k cps", "700k rps", "45 m",
            "layer 4 throughput", "layer 4 connections per second",
            "layer 7 requests per second", "layer 4 concurrent connection",
            "5 virtual instance", "20 virtual instances",
            "role based access control", "rest api",
            "hypervisor", "virtualization", "virtual instance",
            "standalone", "virtualized mode",
            "compression",
        ],
        "title_words": [
            "alteon", "ltm", "load balancer", "adc", "application delivery",
            "application delivery controller",
        ],
        "negative": ["web application firewall", "waf", "ngfw",
                     "next-generation firewall"],
    },

    "IPS": {
        "keywords": [
            # original
            "intrusion prevention", "intrusion detection", "ips", "ids",
            "signature-based detection", "anomaly detection", "snort",
            "suricata", "network-based ips", "nips", "inline ips",
            "zero-day intrusion", "exploit detection",
            "defenseflow", "tipping point",
            # from RFP text
            "ips detection", "ips mitigation",
            "dedicated appliance", "not a part/added functionality",
            "in-line", "span port", "out-of-path",
            "ssl based attack", "behavior based technology",
            "zero-day network-flood", "traffic anomalies",
            "tcp floods", "syn flood", "tcp fin", "tcp reset", "tcp syn",
            "tcp fragmentation", "udp flood", "icmp flood", "igmp flood",
            "dns protection", "a, mx, ptr, aaaa", "text, soa, naptr, srv",
            "syn protection", "safe reset", "tcp reset",
            "dns challenge", "dns rate limit",
            "http challenge response without scripts",
            "sip flood protection", "udp fragmented flood",
            "web vulnerabilities", "mail server vulnerabilities",
            "ftp server vulnerabilities", "sql server vulnerabilities",
            "dns server vulnerabilities", "sip server vulnerabilities",
            "worms and viruses", "trojans and backdoors", "irc bots",
            "spyware", "phishing", "anonymizers",
            "string match engine", "5000+ inbuilt ips signatures",
            "custom signatures",
            "automated ips protection", "dns protections",
            "syn-flood protection", "traffic filters", "anti-scanning profile",
            "port scanning behavioral protection",
            "5 gbps", "10gbps", "attack concurrent sessions",
            "8 x 1 copper ports", "4 x 10g", "bypass",
            "latency should be less than 60 microseconds",
            "dedicated management port",
            "external centralized management",
        ],
        # ── DISAMBIGUATION NOTE ───────────────────────────────────────────────
        # title_words get +3 each and are matched only in the first 200 chars.
        # IPS and DDoS share flood keywords (+1 each, anywhere in chunk).
        # A chunk whose title/header says "Intrusion Prevention" will score
        # IPS; a chunk whose title says "DDoS" / "Anti-DDoS" will score DDoS.
        # The taxonomy classifier picks the SINGLE best-scoring category, so
        # strong title_word hits cleanly break the tie.
        # ─────────────────────────────────────────────────────────────────────
        "title_words": [
            "ips", "intrusion prevention", "ids", "defenseflow",
            "intrusion prevention system",
            "tipping point", "nips",
        ],
        "negative": ["firewall", "waf", "load balancer", "next-generation firewall",
                     "anti-ddos", "ddos mitigation", "ddos protection"],
    },

    "DDoS": {
        "keywords": [
            "ddos", "denial of service", "syn flood", "udp flood",
            "volumetric attack", "scrubbing", "arbor",
            "netscout", "anti-ddos", "ddos mitigation",
            "flood protection", "attack mitigation",
            "ddos protection", "volumetric", "bandwidth attack",
            "amplification attack", "reflection attack",
            "radware defensessl", "alteon ssl",
        ],
        "title_words": [
            "ddos", "anti-ddos", "ddos mitigation", "scrubbing",
            "flood protection", "ddos protection",
        ],
        # DDoS chunks should NOT be confused with generic IPS chunks
        "negative": ["intrusion prevention system", "ips appliance",
                     "nips", "inline ips", "ids"],
    },

    "SWITCH": {
        "keywords": [
            # original
            "network switch", "ethernet switch", "layer 2", "layer 3 switch",
            "vlan", "spanning tree", "stp", "rstp", "lacp", "lldp",
            "catalyst", "nexus", "aruba", "juniper ex",
            "fortiswitch", "802.1q", "port channel",
            # from RFP text
            "network switches", "ten network switches",
            "same os", "same dashboard",
            "vxlan", "evpn", "vxlan+evpn overlay",
            "port acl", "l2, l3 and l4 parameters",
            "1m or higher ipv4 routes",
            "128 ipsec tunnels",
            "gre tunnels",
            "issu", "software upgrades", "hitless patching",
            "graceful insertion and removal", "gir",
            "hot-swappable fans", "hot-swappable power",
            "ac and dc power supply",
            "control plane protection", "copp",
            "port based dos protection", "pdp",
            "bash, python", "openconfig",
            "in-band telemetry",
            "8 queues per port", "priority queue",
            "acl based classification", "qos",
            "policing and shaping",
        ],
        "title_words": [
            "switch", "catalyst", "nexus", "fortiswitch",
            "network switch",
        ],
        "negative": ["firewall", "router", "waf"],
    },

    "ROUTER": {
        "keywords": [
            # original
            "router", "routing", "bgp", "ospf", "mpls", "sd-wan",
            "wan edge", "asr", "juniper mx", "vrf", "route reflector",
            "fortiwan", "silverpeak", "velocloud",
            # from RFP text
            "four routers", "sdwan router", "additional license",
            "ospfv2", "is-is", "ripv2", "bfd", "vrrp v4",
            "nat", "ipv4 routes",
            "ipsec tunnels", "aes256", "encrypted throughput",
            "traditional router", "upgrading to sdwan",
        ],
        "title_words": [
            "router", "sd-wan", "wan", "asr", "mx series",
        ],
        "negative": ["switch", "firewall"],
    },

    "ENDPOINT": {
        "keywords": [
            # original
            "endpoint security", "edr", "endpoint detection",
            "antivirus", "anti-malware", "crowdstrike", "symantec",
            "trend micro", "mcafee", "forticlient", "host-based",
            "endpoint protection", "agent-based",
            # from RFP text
            "windows, linux, redhat", "centos", "oracle", "debian",
            "suse", "ubuntu", "solaris", "aix", "amazon linux",
            "microsoft windows server", "red hat enterprise linux",
            "oracle linux",
            "anti-malware", "host intrusion prevention system",
            "application control", "integrity monitoring", "sandbox integration",
            "boot sector", "master boot sector", "memory resident",
            "macro", "stealth and polymorphism",
            "spyware, adware, dialers", "joke programs", "remote access",
            "hacking tools",
            "dlls", "applications' sub-components",
            "delete, block, quarantine",
            "post infection cleanup", "malicious file",
            "affected registry entries", "windows services created by malware",
            "fileless malware", "command and control", "c&c traffic",
            "memory and boot sector", "real time protection",
            "compressed file formats", "malicious and dangerous websites",
            "machine learning on windows", "behavioral techniques",
            "suspicious files to a sandbox",
            "on-premise sandboxing",
            "port scans", "ipv4/ipv6 attacks",
            "cve details", "hips rules", "network ips rules",
            "cve cross referencing",
            "unauthorized applications", "blacklist/whitelist",
            "integrity monitoring", "registry keys",
            "auto-recommendation scans",
            "local update server",
            "single unified management dashboard",
            "enterprise-wide visibility",
            "granular policies", "distributed environments",
            "central repository", "receive logs",
            "alerts", "security incident",
            "email or snmp traps",
            "multiple administrator", "logging of administrative activities",
            "detailed reports", "export reports to pdf",
            "connected threat defense",
            "suspicious objects", "iocs",
            "antimalware", "file integrity monitoring",
            "central management",
            "ten instances", "ten quantities",
        ],
        "title_words": [
            "endpoint", "edr", "crowdstrike", "forticlient",
            "endpoint security", "endpoint protection",
        ],
        "negative": ["network", "firewall"],
    },

    "APT": {
        "keywords": [
            "advanced persistent threat", "apt",
            "sandbox", "malware analysis", "zero-day",
            "threat intelligence", "advanced malware protection",
            "deep discovery", "sandboxing",
            # from RFP text
            "dynamic real-time analysis", "advanced malware",
            "spear phishing attack", "drive by download",
            "watering hole", "targeted advanced persistent threat",
            "cloud infrastructure system", "analysis and detection of malware",
        ],
        "title_words": ["apt", "sandbox", "deep discovery"],
        "negative": [],
    },

    "ZTNA": {
        "keywords": [
            "ztna", "zero trust network access",
            "secure remote access", "identity-based access",
            "least privilege access",
        ],
        "title_words": ["ztna", "zero trust"],
        "negative": [],
    },

    "DLP": {
        "keywords": [
            "data loss prevention", "dlp",
            "data leakage prevention",
            "sensitive data protection",
            "information protection",
        ],
        "title_words": ["dlp", "data loss prevention"],
        "negative": [],
    },

    "NAC": {
        "keywords": [
            "network access control", "nac",
            "device profiling",
            "guest access",
            "endpoint posture",
            "device visibility",
        ],
        "title_words": ["nac"],
        "negative": [],
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
            "wireless controller",
        ],
        "title_words": ["access point", "wireless", "wifi"],
        "negative": [],
    },

    "OSINT_DARKWEB": {
        "keywords": [
            "dark web",
            "deep web",
            "osint",
            "open source intelligence",
            "threat intelligence feed",
            "threat actor monitoring",
            "credential leak monitoring",
        ],
        "title_words": ["osint", "dark web"],
        "negative": [],
    },

    "STORAGE": {
        "keywords": [
            "storage",
            "san",
            "nas",
            "object storage",
            "block storage",
            "storage array",
        ],
        "title_words": ["storage"],
        "negative": [],
    },

    "HCI": {
        "keywords": [
            "hyperconverged",
            "hyper-converged",
            "hci",
            "virtualized infrastructure",
        ],
        "title_words": ["hci", "hyperconverged"],
        "negative": [],
    },

    "SERVER": {
        "keywords": [
            "server",
            "blade server",
            "rack server",
            "compute node",
            "poweredge",
            "proliant",
        ],
        "title_words": ["server"],
        "negative": [],
    },

    "SERVER_RACK": {
        # New category derived from the RFP (42U rack spec)
        "keywords": [
            "42u rack", "server rack",
            "mounting rails", "cage nuts",
            "height: 73.5 inches", "1867 mm",
            "19 inches", "482.6 mm",
            "perforated", "airflow and cooling",
            "cable management", "cable channels", "cable tie points",
            "grounding and bonding",
            "casters",
            "rack units",
            "19-inch rack standard",
            "power distribution units", "pdu",
            "front and rear doors",
            "removable side panels",
            "weight capacity",
            "one unit 42u",
        ],
        "title_words": ["42u rack", "server rack", "rack"],
        "negative": ["firewall", "router", "switch"],
    },

    "PACKET_CAPTURE_PROBE": {
        # New category derived from the RFP (network traffic capture / probe)
        "keywords": [
            "network traffic capture",
            "pcap",
            "packet capture",
            "100 gb/s without any packet loss",
            "500 mbps",
            "probe",
            "collector",
            "capture requests",
            "monitoring interfaces",
            "custom filtering criteria",
            "mac addresses, vlan tags",
            "mpls labels",
            "ipv4, ipv6 addresses",
            "tcp, udp and sctp ports",
            "voip calls", "sip or h.323",
            "time interval of traffic recording",
            "automatic rotation of old data",
            "short term full packet history",
            "netflow analysis",
            "event tree",
            "expert knowledge",
            "remedial actions",
            "api for retrieving records",
            "capture time period",
            "4x 1/10g sfp interfaces",
            "2*10g sr sfp", "2*10g lr sfp",
            "analysis results",
            "non-standard states",
        ],
        "title_words": [
            "packet capture", "probe", "traffic capture",
            "network capture",
        ],
        "negative": ["firewall", "load balancer"],
    },

    "CLOUD_SERVICES": {
        "keywords": [
            "cloud service",
            "iaas",
            "paas",
            "saas",
            "cloud-native",
            "cloud infrastructure",
        ],
        "title_words": ["cloud"],
        "negative": [],
    },

    "CSPM_CWPP": {
        "keywords": [
            "cspm",
            "cwpp",
            "cloud posture",
            "cloud workload protection",
            "container security",
            "runtime protection",
        ],
        "title_words": ["cspm", "cwpp"],
        "negative": [],
    },

    "OT_SECURITY": {
        "keywords": [
            "ot security",
            "industrial security",
            "ics",
            "scada",
            "operational technology",
        ],
        "title_words": ["ot", "scada"],
        "negative": [],
    },

    "PIM_PAM": {
        "keywords": [
            "pam",
            "pim",
            "privileged access management",
            "privileged identity management",
            "credential vault",
        ],
        "title_words": ["pam", "pim"],
        "negative": [],
    },

    "SSL_CERTIFICATE": {
        "keywords": [
            "ssl certificate",
            "tls certificate",
            "certificate lifecycle",
            "certificate authority",
        ],
        "title_words": ["certificate"],
        "negative": [],
    },

    "KEY_MANAGEMENT_HSM": {
        "keywords": [
            "hsm",
            "hardware security module",
            "key management",
            "cryptographic key",
        ],
        "title_words": ["hsm"],
        "negative": [],
    },

    "LOG_MANAGEMENT": {
        "keywords": [
            "log management",
            "syslog",
            "event logging",
            "log analytics",
        ],
        "title_words": ["log"],
        "negative": ["siem"],
    },

    "APPLICATION_PERFORMANCE_MONITORING_SEARCH": {
        "keywords": [
            "application performance monitoring",
            "apm",
            "observability",
            "distributed tracing",
            "elasticsearch",
            "search platform",
        ],
        "title_words": ["apm", "observability"],
        "negative": [],
    },

    "SIEM": {
        "keywords": [
            "siem",
            "security information and event management",
            "security analytics",
            "event correlation",
            "ueba",
            "soc",
            "threat hunting",
        ],
        "title_words": ["siem"],
        "negative": [],
    },
}