from typing import Dict, List, Callable, Any

from overrides import overrides

from allennlp.state_machines.states import GrammarStatelet
from allennlp.state_machines.states.grammar_statelet import ActionRepresentation


class RnnGrammarStatelet(GrammarStatelet):
    """
    In addition to the non-terminal stack, this statelet also keeps track of the information from an RNN that is
    encoding the tree.
    """
    def __init__(self,
                 nonterminal_stack: List[str],
                 rnn_state_stack: List[Any],
                 valid_actions: Dict[str, ActionRepresentation],
                 is_nonterminal: Callable[[str], bool],
                 reverse_productions: bool = True) -> None:
        super().__init__(nonterminal_stack,
                         valid_actions,
                         is_nonterminal,
                         reverse_productions)
        self._rnn_state_stack = rnn_state_stack

    @overrides
    def take_action(self,  # pylint: disable=arguments-differ
                    production_rule: str,
                    rnn_state: Any) -> 'RnnGrammarStatelet':
        """
        For each nonterminal being pushed to the nonterminal stack, we push the ``rnn_state`` to the
        ``rnn_state_stack``. The idea is that ``rnn_state`` is the state that caused the prediction of the rule,
        whose right side contains the new nonterminals. When each of them is being expanded, the tree-encoder will
        want to use this RNN state as the input to the RNN cell. If the right side is a terminal, the ``rnn_state``
        will not be stored.
        """
        nonterminal_stack = super().take_action(production_rule)._nonterminal_stack  # pylint: disable=protected-access
        rnn_state_stack = self._rnn_state_stack[:-1]

        _, right_side = production_rule.split(' -> ')
        productions = self._get_productions_from_string(right_side)
        if self._reverse_productions:
            productions = list(reversed(productions))
        for production in productions:
            if self._is_nonterminal(production):
                rnn_state_stack.append(rnn_state)

        return RnnGrammarStatelet(nonterminal_stack=nonterminal_stack,
                                  rnn_state_stack=rnn_state_stack,
                                  valid_actions=self._valid_actions,
                                  is_nonterminal=self._is_nonterminal,
                                  reverse_productions=self._reverse_productions)

    def get_parent_rnn_state(self) -> Any:
        """
        Returns the top of the ``rnn_state_stack``.
        """
        return self._rnn_state_stack[-1]
