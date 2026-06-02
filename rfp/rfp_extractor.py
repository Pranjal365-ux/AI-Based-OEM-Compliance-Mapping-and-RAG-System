# rfp/rfp_extractor.py

import json
import logging
import re
from pathlib import Path
from typing import List

from ingestion.pdf_extractor import PDFExtractor
from models.schemas import Requirement
from services.llm_services import llm

logger = logging.getLogger(__name__)


class RFPRequirementExtractor:
    """
    Extracts structured requirements from RFP documents.

    Strategy:
    1. Extract PDF text
    2. Split into sections
    3. Regex extraction for quantitative requirements
    4. LLM extraction for qualitative requirements
    5. Normalize + validate
    """

    CATEGORY_KEYWORDS = {
    "Firewall": [
        "firewall", "ngfw", "utm", "threat prevention",
        "ssl inspection", "ips", "ids", "sd-wan"
    ],

    "SIEM": [
        "siem", "soc", "ueba", "event correlation",
        "log management", "security analytics",
        "security monitoring"
    ],

    "EDR": [
        "edr", "endpoint detection", "endpoint protection",
        "xdr", "anti-malware", "ransomware protection"
    ],

    "NDR": [
        "ndr", "network detection", "network analytics"
    ],

    "SOAR": [
        "soar", "orchestration", "automated response",
        "playbook"
    ],

    "Email Security": [
        "email security", "secure email gateway",
        "anti phishing", "spam filtering"
    ],

    "WAF": [
        "web application firewall",
        "waf",
        "owasp"
    ],

    "Load Balancer": [
        "load balancer",
        "application delivery controller",
        "adc"
    ],

    "VPN": [
        "vpn", "ipsec", "ssl vpn",
        "remote access"
    ],

    "Router": [
        "router", "routing",
        "bgp", "ospf"
    ],

    "Switch": [
        "switch", "ethernet",
        "layer 2", "layer 3"
    ],

    "Wireless": [
        "wireless", "wifi",
        "access point", "802.11"
    ],

    "NAC": [
        "nac", "network access control"
    ],

    "PAM": [
        "pam", "privileged access"
    ],

    "IAM": [
        "identity management",
        "iam",
        "sso",
        "mfa"
    ],

    "DLP": [
        "dlp",
        "data loss prevention"
    ],

    "Cloud Security": [
        "casb",
        "cwpp",
        "cnapp",
        "cloud security"
    ],
}

    QUANT_PATTERN = re.compile(
        r"(?P<metric>.+?)"
        r"(?:>=|=>|at least|minimum|min\.?)\s*"
        r"(?P<value>\d+(?:\.\d+)?)\s*"
        r"(?P<unit>Gbps|Mbps|TB|GB|MB|Users|Sessions|EPS)",
        re.IGNORECASE,
    )

    def __init__(self):
        self.pdf_extractor = PDFExtractor()

    # --------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------

    def extract_from_pdf(self, pdf_path: str) -> List[Requirement]:

        logger.info(f"Processing RFP: {pdf_path}")

        extracted = self.pdf_extractor.extract(pdf_path)

        full_text = extracted.text

        return self.extract_requirements(full_text)

    def extract_requirements(self, text: str) -> List[Requirement]:

        sections = self.segment_sections(text)

        requirements = []

        req_counter = 1

        for section_name, section_text in sections:

            regex_reqs = self._extract_quantitative(
                section_text,
                section_name,
                req_counter,
            )

            requirements.extend(regex_reqs)

            req_counter += len(regex_reqs)

            llm_reqs = self._extract_qualitative(
                section_text,
                section_name,
                req_counter,
            )

            requirements.extend(llm_reqs)

            req_counter += len(llm_reqs)

        return self._deduplicate(requirements)

    # --------------------------------------------------
    # SECTION SPLITTING
    # --------------------------------------------------

    def segment_sections(self, text: str):

        sections = []

        current_title = "General"

        current_lines = []

        for line in text.splitlines():

            line = line.strip()

            if not line:
                continue

            if self._looks_like_heading(line):

                if current_lines:
                    sections.append(
                        (
                            current_title,
                            "\n".join(current_lines),
                        )
                    )

                current_title = line
                current_lines = []

            else:
                current_lines.append(line)

        if current_lines:
            sections.append(
                (
                    current_title,
                    "\n".join(current_lines),
                )
            )

        return sections

    # --------------------------------------------------
    # QUANTITATIVE EXTRACTION
    # --------------------------------------------------

    def _extract_quantitative(
        self,
        text: str,
        section: str,
        start_idx: int,
    ):

        results = []

        counter = start_idx

        for sentence in self._split_sentences(text):

            match = self.QUANT_PATTERN.search(sentence)

            if not match:
                continue

            metric = match.group("metric").strip()

            results.append(
                Requirement(
                    requirement_id=f"REQ-{counter:04}",
                    category=self._detect_category(sentence),
                    requirement=metric,
                    source_text=sentence,
                    mandatory=self._is_mandatory(sentence),
                    operator=">=",
                    value=match.group("value"),
                    unit=match.group("unit"),
                    section=section,
                )
            )

            counter += 1

        return results

    # --------------------------------------------------
    # QUALITATIVE EXTRACTION
    # --------------------------------------------------

    def _extract_qualitative(
        self,
        text: str,
        section: str,
        start_idx: int,
    ):

        prompt = f"""
Extract cybersecurity requirements.

Return JSON only.

Format:

[
 {{
   "requirement":"...",
   "category":"...",
   "mandatory":true,
   "source_text":"..."
 }}
]

TEXT:
{text[:6000]}
"""

        try:

            response = llm.generate(prompt)

            data = json.loads(response)

            results = []

            counter = start_idx

            for item in data:

                results.append(
                    Requirement(
                        requirement_id=f"REQ-{counter:04}",
                        category=item.get(
                            "category",
                            "Other",
                        ),
                        requirement=item["requirement"],
                        source_text=item["source_text"],
                        mandatory=item.get(
                            "mandatory",
                            True,
                        ),
                        operator="supports",
                        value="true",
                        unit=None,
                        section=section,
                    )
                )

                counter += 1

            return results

        except Exception as exc:

            logger.warning(
                f"LLM extraction failed: {exc}"
            )

            return []

    # --------------------------------------------------
    # HELPERS
    # --------------------------------------------------

    def _detect_category(self, text: str) -> str:

        lower = text.lower()

        for category, keywords in self.CATEGORY_KEYWORDS.items():

            for kw in keywords:

                if kw in lower:
                    return category

        return self._llm_detect_category(text)
    
    def _llm_detect_category(self, text: str) -> str:

        prompt = f"""
            Classify this cybersecurity requirement based on the product that the requirement is asking for.

            Return ONLY one category.

            Requirement:
            {text}
            """

        try:
            result = llm.generate(
                prompt,
                temperature=0,
                max_tokens=20,
            )

            return result.strip()

        except Exception:
            return "Other"

    def _is_mandatory(self, text: str) -> bool:

        text = text.lower()

        mandatory_words = [
            "shall",
            "must",
            "mandatory",
            "required",
        ]

        optional_words = [
            "should",
            "preferred",
            "optional",
        ]

        if any(word in text for word in mandatory_words):
            return True

        if any(word in text for word in optional_words):
            return False

        return True

    def _split_sentences(self, text: str):

        return re.split(
            r"(?<=[.!?])\s+",
            text,
        )

    def _looks_like_heading(self, line: str):

        if len(line) > 80:
            return False

        if re.match(
            r"^\d+(\.\d+)*\s+",
            line,
        ):
            return True

        if line.isupper():
            return True

        return False

    def _deduplicate(
        self,
        requirements: List[Requirement],
    ):

        seen = set()

        unique = []

        for req in requirements:

            key = (
                req.requirement.lower(),
                req.category,
            )

            if key not in seen:

                seen.add(key)

                unique.append(req)

        return unique