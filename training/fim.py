"""Fill-in-the-Middle (FIM) training transforms."""

import random

# These will be resolved to token strings at usage time
FIM_PREFIX = "<|fim_prefix|>"
FIM_MIDDLE = "<|fim_middle|>"
FIM_SUFFIX = "<|fim_suffix|>"


def psm_transform(text: str, fim_rate: float = 0.5) -> str | None:
    """Prefix-Suffix-Middle transform.
    With probability fim_rate, rearrange text as: PREFIX prefix SUFFIX suffix MIDDLE middle
    Returns None if transform not applied (original text should be used)."""
    if random.random() > fim_rate:
        return None
    # Pick a random split point
    if len(text) < 10:
        return None
    split = random.randint(1, len(text) - 1)
    prefix = text[:split]
    middle = text[split:]
    return f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{FIM_MIDDLE}{middle}"


def spm_transform(text: str, fim_rate: float = 0.5) -> str | None:
    """Suffix-Prefix-Middle transform.
    Same as PSM but suffix comes before prefix."""
    if random.random() > fim_rate:
        return None
    if len(text) < 10:
        return None
    split = random.randint(1, len(text) - 1)
    prefix = text[:split]
    suffix = text[split:]  # NOTE: suffix is the part AFTER split
    middle = ""  # In SPM, middle is what goes between prefix and suffix
    # Actually for FIM: prefix is before cursor, suffix is after cursor, middle is what to fill
    # Let's do it correctly:
    # Choose a span to mask
    start = random.randint(0, len(text) - 2)
    end = random.randint(start + 1, len(text))
    prefix = text[:start]
    middle = text[start:end]
    suffix = text[end:]
    return f"{FIM_SUFFIX}{suffix}{FIM_PREFIX}{prefix}{FIM_MIDDLE}{middle}"


def apply_fim_augmentation(
    texts: list[str],
    fim_rate: float = 0.5,
    psm_ratio: float = 0.5,
) -> list[str]:
    """Apply FIM transforms to a batch of texts.

    Args:
        texts: List of code strings
        fim_rate: Probability of applying FIM to each text
        psm_ratio: Ratio of PSM vs SPM transforms

    Returns:
        List of (possibly transformed) texts
    """
    results = []
    for text in texts:
        if random.random() < psm_ratio:
            transformed = psm_transform(text, fim_rate)
        else:
            transformed = spm_transform(text, fim_rate)
        results.append(transformed if transformed is not None else text)
    return results
