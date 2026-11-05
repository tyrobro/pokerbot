"""confidence_weight(k) (Section 4.1): level 1 has confidence_weight = 0
(pure baseline, no deviation ever); level 5 has the highest deviation
ceiling. Linear default (k/5), per Section 4.1's own stated default --
"pick linear as the default and revisit only if calibration results
look wrong" -- not independently re-derived here."""


def confidence_weight(ml_level: int) -> float:
    if not 1 <= ml_level <= 5:
        raise ValueError("ml_level must be 1-5")
    if ml_level == 1:
        return 0.0
    return ml_level / 5.0