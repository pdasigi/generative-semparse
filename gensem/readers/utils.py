from typing import List, Callable, Tuple


def get_nonterminals_from_list(right_side: str) -> List[str]:
    return right_side[1:-1].split(', ')  # "[A, B]" -> ["A", "B"]


def make_bottom_up_action_sequence(top_down_sequence: List[str],
                                   is_nonterminal: Callable[[str], bool]) -> List[str]:
    """
    Takes a top-down sequence of production rules and returns a bottom-up sequence. The top-down sequence is
    expected to be depth-first, and the right sides of all productions are expected to be all non-terminals or a
    single terminal. We process the actions linearly, and produce terminal productions as we see them, and
    non-terminal productions as soon as all the terminal productions corresponding to their right sides have been
    produced.
    """
    nonterminal_stack: List[Tuple[str, str]] = [('@start@', None)]  # list of (nonterminal, production/None)
    bottom_up_sequence: List[str] = []
    for action in top_down_sequence:
        left_side, right_side = action.split(' -> ')
        # Keep popping until we see the left side at the top of the stack.
        while nonterminal_stack[-1][0] != left_side:
            next_action = nonterminal_stack[-1][1]
            assert next_action is not None
            bottom_up_sequence.append(next_action)
            nonterminal_stack = nonterminal_stack[:-1]
        # We see the left side at the top now. We pop it.
        nonterminal_stack = nonterminal_stack[:-1]
        #Now we can process the right side.
        if not is_nonterminal(right_side) and not right_side[0] == '[':
            # We output the current action. It does not need to go on the stack.
            bottom_up_sequence.append(action)
        else:
            if right_side[0] == '[':
                right_side_list = get_nonterminals_from_list(right_side)
            else:
                right_side_list = [right_side]
            # We push the action on the stack as an expanded action.
            nonterminal_stack.append((left_side, action))
            # And the nonterminals as actions to be expanded.
            for symbol in reversed(right_side_list):
                nonterminal_stack.append((symbol, None))

    for _, action in reversed(nonterminal_stack):
        assert action is not None
        bottom_up_sequence.append(action)

    return bottom_up_sequence
