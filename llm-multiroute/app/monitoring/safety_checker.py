import re

# Common prompt injection patterns
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)", re.IGNORECASE),
    re.compile(r"forget\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+", re.IGNORECASE),
    re.compile(r"act\s+as\s+(if\s+you\s+are|a)\s+", re.IGNORECASE),
    re.compile(r"pretend\s+(you\s+are|to\s+be)\s+", re.IGNORECASE),
    re.compile(r"(reveal|show|print|output)\s+(your\s+)?(system\s+prompt|instructions|rules)", re.IGNORECASE),
    re.compile(r"what\s+(are|is)\s+your\s+(system\s+prompt|instructions|rules)", re.IGNORECASE),
    re.compile(r"\bsystem\s*:\s*", re.IGNORECASE),
    re.compile(r"<\s*system\s*>", re.IGNORECASE),
    re.compile(r"\[INST\]", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"DAN\s+mode", re.IGNORECASE),
    re.compile(r"do\s+anything\s+now", re.IGNORECASE),
]

# Policy violation patterns (harmful content indicators)
_POLICY_PATTERNS = [
    re.compile(r"how\s+to\s+(make|build|create)\s+(a\s+)?(bomb|weapon|explosive)", re.IGNORECASE),
    re.compile(r"(generate|create|write)\s+(malware|virus|ransomware)", re.IGNORECASE),
    re.compile(r"(hack|breach|exploit)\s+(into|a)\s+", re.IGNORECASE),
]


class SafetyChecker:
    def check_prompt_injection(self, text: str) -> dict:
        """Check text for prompt injection attempts.

        Returns dict with 'detected' bool and 'patterns' list of matched pattern descriptions.
        """
        matched = []
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(text):
                matched.append(pattern.pattern)

        return {
            "detected": len(matched) > 0,
            "patterns": matched,
        }

    def check_policy_violations(self, text: str, response: str) -> dict:
        """Check input and output text for policy violation patterns.

        Returns dict with 'detected' bool, 'input_matches' and 'output_matches'.
        """
        input_matches = []
        output_matches = []

        for pattern in _POLICY_PATTERNS:
            if pattern.search(text):
                input_matches.append(pattern.pattern)
            if pattern.search(response):
                output_matches.append(pattern.pattern)

        return {
            "detected": len(input_matches) > 0 or len(output_matches) > 0,
            "input_matches": input_matches,
            "output_matches": output_matches,
        }
