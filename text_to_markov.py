from collections import Counter
from itertools import accumulate, islice
import networkx as nx
import random
import re
from typing import List, Optional, Sequence, Tuple

# add some attributes capturing statistics (avg degree in and out, etc)
# maybe some visualization stuff, deal with terminal nodes
# maybe add warm starts
# ellipses?

# NB: there are terminal nodes: this is not a Markov chain.
# Maybe implement something like <END> nodes.

PUNCTUATION = '[.!?,:;]'
TERMINAL_PUNCTUATION = '[.!?]'


class TextMarkov():
    def __init__(self,
                 n_grams: int = 1,
                 unigram_regex: str = r"(?u)[\w']+",
                 tokenize_punctuation: bool = True) -> None:
        self.n_grams = n_grams
        if tokenize_punctuation:
            self.unigram_regex = unigram_regex + '|' + PUNCTUATION
        else:
            self.unigram_regex = unigram_regex
        self.markov_chain = nx.DiGraph()

    def fit(self, text: str) -> nx.DiGraph:
        tokens = _tokenize(self.n_grams, self.unigram_regex, text)
        token_edges = zip(tokens, tokens[self.n_grams:])
        self.tokens = set(tokens)

        # Put the multiset Counter(token_edges) in a form that
        # add_weighted_edges_from accepts.
        edges_freqs = [(*edge, frequency)
                       for edge, frequency in Counter(token_edges).items()]

        self.markov_chain.add_nodes_from(self.tokens)
        self.markov_chain.add_weighted_edges_from(edges_freqs)

        total_freqs = {n_gram: self.markov_chain.out_degree(n_gram,
                                                            weight='weight')
                       for n_gram in self.markov_chain}
        for n_gram, _, labels in self.markov_chain.edges(data=True):
            labels['weight'] = labels['weight'] / total_freqs[n_gram]

        return self.markov_chain

    def generate_tokens(self, start: str = ''):
        if not self.markov_chain:
            raise NotFittedError(f'This {type(self).__name__} instance has '
                                 'not been fitted yet. Try calling "fit" '
                                 'first.')
        filtered_tokens = {token 
                           for token in self.tokens 
                           if re.match('^' + start, token)}
        current_token, = random.sample(filtered_tokens, 1)

        while True:
            yield current_token
            weighted_edges = self.markov_chain.out_edges(
                    current_token,
                    data=True)
            if weighted_edges:
                _, next_tokens, data = zip(*weighted_edges)
                probabilities = [datum['weight']
                                 for datum in data]
                current_token, = random.choices(next_tokens, weights=probabilities)
            else:
                raise TerminalNodeError('Terminal node reached')

    def generate_sentence(self,
                          start_token: Optional[Tuple[str]] = None,
                          max_tokens: int = 100) -> str:
        sentence_tokens = take_until_inclusive(
                lambda tk: any(re.match(TERMINAL_PUNCTUATION, uni) for uni in tk),
                self.generate_tokens(start_token=start_token))
        big_n_gram = flatten(sentence_tokens)
        output = n_gram_to_string(big_n_gram)
        terminal_regex = fr'(?<={TERMINAL_PUNCTUATION})[\s\S]*$'
        return re.sub(terminal_regex, '', output)

    def generate_text(self,
                      n_tokens: int = 30,
                      start_token: Optional[Tuple[str]] = None) -> str:
        generated_tokens = islice(
                self.generate_tokens(start_token=start_token),
                n_tokens)
        big_n_gram = flatten(generated_tokens)
        return n_gram_to_string(big_n_gram)


def _tokenize(n_gram: int, unigram_regex: str, text: str) -> List[Tuple[str]]:
    unigrams = re.findall(unigram_regex, text)
    return [concatenate_grams(tuple_gram) 
            for tuple_gram in zip(*(unigrams[i:] for i in range(n_gram)))]


def n_gram_to_string(n_gram: str) -> str:
    output = n_gram[0]
    for unigram in n_gram[1:]:
        if re.match(PUNCTUATION, unigram):
            output = output + unigram
        else:
            output = output + ' ' + unigram
    return output


def take_until_inclusive(predicate, iterator):
    for x in iterator:
        yield x
        if predicate(x):
            break


def concatenate_two_grams(gram1: str, gram2: str) -> str:
    if re.match(PUNCTUATION, gram2[0]):
        return gram1 + gram2
    else:
        return gram1 + ' ' + gram2


def concatenate_grams(grams: Sequence[str]) -> str:
    return list(accumulate(grams, concatenate_two_grams))[-1]


def flatten(sequence_of_tuples):
    return sum(sequence_of_tuples, ())


class NotFittedError(Exception):
    pass


class TerminalNodeError(Exception):
    pass
