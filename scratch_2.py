def identify_matches(i, j, Qx0, Qy0, Qx1, Qy1, Rx0, Ry0, Rx1, Ry1, min_length_cqt):

    # get indices corresponding to query(Q)
    Q_indices = set(range(Qx0, Qx1+1))

    # get indices corresponding to query(Q)
    R_indices = set(range(Rx0, Rx1+1))

    # query on the left
    if Qx0 <= Rx0:
        # indices in common between query(Q) and returned(R)
        overlap_indices = Q_indices.intersection(R_indices)
        overlap_sig = len(overlap_indices) >= min_length_cqt
    # query on the right
    else:
        # indices in common between query(Q) and returned(R)
        overlap_indices = R_indices.intersection(Q_indices)
        overlap_sig = len(overlap_indices) >= min_length_cqt

    if overlap_sig:
        update_dict(match_dict, i, j)
        update_dict(match_dict, j, i)

    return match_dict

