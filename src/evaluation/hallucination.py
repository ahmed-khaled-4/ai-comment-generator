# src/evaluation/hallucination.py
import re

class HallucinationDetector:
    def analyze(self, code, comment, score_metrics=None):
        """
        Analyzes code and comment to detect specific hallucination types.
        Returns a list of flags/warnings.
        """
        flags = []
        
        # 1. Check for "Generic" comments (low information)
        generic_phrases = ["this function does", "function to", "code snippet", "todo"]
        if len(comment.split()) < 5:
            flags.append("TOO_SHORT")
        elif any(phrase in comment.lower() for phrase in generic_phrases) and len(comment.split()) < 10:
            flags.append("GENERIC_RESPONSE")

        # 2. Variable Hallucination Check (Did it invent parameters?)
        # Simple regex to find python function arguments: def func(arg1, arg2):
        try:
            # Find definition line
            def_match = re.search(r"def\s+\w+\s*\((.*?)\)", code)
            if def_match:
                args_str = def_match.group(1)
                # Extract clean arg names (ignore type hints like :int or default values =None)
                args = [re.split(r"[:=]", arg.strip())[0] for arg in args_str.split(',') if arg.strip()]
                
                # Check if comment mentions "args" that aren't in the function
                # (This is a simplified check; in a real scenario, you'd use an AST)
                # For now, we just look for obvious "param x" patterns in comment
                comment_params = re.findall(r"param\s+(\w+)", comment)
                for param in comment_params:
                    if param not in args and param != 'return':
                        flags.append(f"HALLUCINATION_UNKNOWN_PARAM_({param})")
        except Exception:
            pass # fallback if regex fails

        # 3. Score-based Classification (if metrics provided)
        if score_metrics:
            if score_metrics.get('bertscore_f1', 1.0) < 0.6:
                flags.append("SEMANTIC_MISMATCH")
            if score_metrics.get('bleu', 1.0) < 0.1:
                flags.append("LOW_TEXT_OVERLAP")

        return flags if flags else ["PASS"]