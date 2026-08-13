from mce.align import DELETE, EQUAL, INSERT, SUB, align, edit_distance


def test_identical_sequences_are_all_equal():
    ops = align(list("abc"), list("abc"))
    assert [o.op for o in ops] == [EQUAL, EQUAL, EQUAL]
    assert edit_distance(list("abc"), list("abc")) == 0


def test_substitution_deletion_insertion_are_detected():
    assert edit_distance(list("abc"), list("axc")) == 1
    assert edit_distance(list("abc"), list("ac")) == 1
    assert edit_distance(list("ac"), list("abc")) == 1


def test_ops_carry_reference_indices():
    ops = align(["a", "b", "c"], ["a", "x", "c"])
    sub = [o for o in ops if o.op == SUB][0]
    assert sub.ref_idx == 1
    assert sub.hyp_idx == 1


def test_insertion_is_anchored_at_the_reference_position_it_precedes():
    ops = align(["a", "c"], ["a", "b", "c"])
    ins = [o for o in ops if o.op == INSERT][0]
    # 'b' sits before reference token index 1 ('c')
    assert ins.ref_pos == 1
    assert ins.ref_idx is None


def test_trailing_insertions_anchor_past_the_end():
    ops = align(["a"], ["a", "b", "c"])
    ins = [o for o in ops if o.op == INSERT]
    assert len(ins) == 2
    assert all(o.ref_pos == 1 for o in ins)


def test_empty_reference_is_all_insertions():
    ops = align([], ["a", "b"])
    assert [o.op for o in ops] == [INSERT, INSERT]


def test_empty_hypothesis_is_all_deletions():
    ops = align(["a", "b"], [])
    assert [o.op for o in ops] == [DELETE, DELETE]


def test_reference_positions_are_monotonic():
    ops = align(list("abcde"), list("axcqqe"))
    positions = [o.ref_pos for o in ops]
    assert positions == sorted(positions)
