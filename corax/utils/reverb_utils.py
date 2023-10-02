import sys
from typing import Sequence, Tuple

import reverb


def disable_insert_blocking(
    tables: Sequence[reverb.Table],
) -> Tuple[Sequence[reverb.Table], Sequence[int]]:
    """Disables blocking of insert operations for a given collection of tables."""
    modified_tables = []
    sample_sizes = []
    for table in tables:
        rate_limiter_info = table.info.rate_limiter_info
        rate_limiter = reverb.rate_limiters.RateLimiter(
            samples_per_insert=rate_limiter_info.samples_per_insert,
            min_size_to_sample=rate_limiter_info.min_size_to_sample,
            min_diff=rate_limiter_info.min_diff,
            max_diff=sys.float_info.max,
        )
        modified_tables.append(table.replace(rate_limiter=rate_limiter))
        # Target the middle of the rate limiter's insert-sample balance window.
        sample_sizes.append(
            max(1, int((rate_limiter_info.max_diff - rate_limiter_info.min_diff) / 2))
        )
    return modified_tables, sample_sizes
