from typing import List, Callable, Any, Tuple, Optional

from gensem.readers import utils as reader_utils


class RnnGrammarState:
    """
    Useful while encoding a tree, this statelet keeps track of the yet to be expanded nonterminals, and their
    corresponding RNN states. Note that unlike the GrammarStatelet in allennlp that processes actions top-down,
    this one assumes the action sequence is in bottom-up order. Also, since we use this statelet for encoding, and
    not decoding the tree, we do not need to keep a list of valid actions.
    """
    def __init__(self,
                 nonterminal_stack: List[Tuple[str, Any]],
                 is_nonterminal: Callable[[str], bool]) -> None:
        self._nonterminal_stack = nonterminal_stack
        self._is_nonterminal = is_nonterminal

    def _get_nonterminals_from_rule(self, production_rule: str) -> Tuple[str, List[str]]:
        """
        Returns nonterminals on left and right sides of the rule. If rule is a terminal production, second element
        in the returned tuple will be an empty list.
        """
        left_side, right_side = production_rule.split(' -> ')
        right_nonterminals = []
        if right_side[0] == '[':
            right_nonterminals = reader_utils.get_nonterminals_from_list(right_side)
        elif self._is_nonterminal(right_side):
            right_nonterminals = [right_side]
        return left_side, right_nonterminals

    def update(self,
               production_rule: str,
               rnn_state: Any) -> 'RnnGrammarState':
        """
        We push to the nonterminal stack the new production rule and the state of an RNN after processing it, after
        optionally popping any open nonterminals the current production rule closes. Returns a new state with an
        updated stack.
        """
        left_side, nonterminals_to_close = self._get_nonterminals_from_rule(production_rule)

        if nonterminals_to_close:
            assert len(self._nonterminal_stack) >= len(nonterminals_to_close), \
                    f"Too many nonterminals! Cannot update with {production_rule}"
            # We need to do some popping. But we make sure the production is valid first.
            num_nonterminals = len(nonterminals_to_close)
            stack_nonterminals = [x[0] for x in self._nonterminal_stack[-num_nonterminals:]]
            assert stack_nonterminals == nonterminals_to_close, \
                   f"Mismatched nonterminals! Rule is {production_rule}. Stack has {stack_nonterminals}"
            new_stack = self._nonterminal_stack[:-len(nonterminals_to_close)]
        else:
            new_stack = list(self._nonterminal_stack)

        new_stack.append((left_side, rnn_state))

        return RnnGrammarState(nonterminal_stack=new_stack,
                               is_nonterminal=self._is_nonterminal)

    def get_child_rnn_states(self, production_rule: str) -> Optional[List[Any]]:
        """
        Returns the RNN states corresponding to the nonterminals this production rule closes, if it is a
        nonterminal production. If this production rule is a terminal production, will return None.
        """
        _, right_nonterminals = self._get_nonterminals_from_rule(production_rule)
        if not right_nonterminals:
            return None

        assert len(self._nonterminal_stack) >= len(right_nonterminals), \
                f"Too many nonterminals! Cannot get child rnn states for {production_rule}"
        child_rnn_states = []
        for i, nonterminal in enumerate(reversed(right_nonterminals)):
            stack_info = self._nonterminal_stack[-(i+1)]
            assert nonterminal == stack_info[0], \
                    f"Mismatched nonterminals! {production_rule} had {nonterminal}, expected {stack_info[0]}"
            child_rnn_states.append(stack_info[1])
        return child_rnn_states
