from collections import Counter
from functools import partial
from itertools import accumulate, islice
import networkx as nx
import random
import re
from typing import Callable, Generator, Iterator, List, Optional, Sequence, TypeVar

PUNCTUATION = '[.!?,:;]'
TERMINAL_PUNCTUATION = '[.!?]'


class TextMarkov():
    '''
    Markov text generator.

    TextMarkov takes a string, and creates a Markov chain whose nodes
    are the tokens (for example words, or pairs of consecutive words)
    in this string, and in which the transition probability to go from
    one token to another is the probability with which this occurs in
    the given string. The usage is similar to the sklearn API. That
    is, the model should first be fitted using a call to the "fit"
    method, and then it can be used as a Markov text generator by
    calling the "generate_text" and "generate_sentence" methods.

    Parameters
    ----------
    n_grams: int, default=1
        Number of unigrams that a single token consists of. If set to
        1, a token will just be a single word or punctuation mark.

    unigram_regex: str
        Regular expression encoding what constitutes a non-punctuation
        unigram. The default selects one or more alphanumeric
        characters or apostrophes.

    tokenize_punctuation: bool, default=True
        If True, punctuation marks will be treated as unigrams.

    Attributes
    ----------
    markov_chain: networkx.Digraph
        The underlying Markov chain of the model. The nodes are the
        tokens occuring in the text on which the Markov chain is based
        The transition probabilities are stored in the weights of the
        edges.

    tokens: Set[str]
        The set of tokens of the Markov chain.
    '''
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
        """
        Fit the Markov chain.

        Parameters
        ----------
        text : str
            The text on which to fit the Markov chain.

        Returns
        -------
        self : returns an instance of self
        """
        tokens = tokenize(self.n_grams, self.unigram_regex, text)
        token_edges = zip(tokens, tokens[self.n_grams:])
        self.tokens = set(tokens)

        # Put the multiset Counter(token_edges) in a form that
        # add_weighted_edges_from accepts.
        edges_freqs = ((*edge, frequency)
                       for edge, frequency in Counter(token_edges).items())

        self.markov_chain.add_nodes_from(self.tokens)
        self.markov_chain.add_weighted_edges_from(edges_freqs)

        total_freqs = {n_gram: self.markov_chain.out_degree(n_gram,
                                                            weight='weight')
                       for n_gram in self.markov_chain}
        for n_gram, _, labels in self.markov_chain.edges(data=True):
            labels['weight'] = labels['weight'] / total_freqs[n_gram]

        return self

    def generate_tokens(self, start: str = '') -> Generator[str, None, None]:
        """
        Make a generator that returns the nodes in a traversal of the
        Markov chain.

        Parameters
        ----------
        start : str, default=''
            Text that the first node in the traversal should start
            with. If empty, the first node will be chosen randomly.

        Returns
        -------
        iterator iterating over strings
        """
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
                yield '<END OF STRING REACHED>'
                return

    def generate_sentence(self,
                          start: str = '',
                          max_tokens: Optional[int] = 100) -> str:
        """
        Generate a single sentence traversing the Markov chain.

        Parameters
        ----------
        start : str, default=''
            Text that the sentence should start with. If empty, the sentence
            will start at a random token of the text.

        max_tokens : int or nonetype, default=100
            The maximum number of tokens to search before giving up.
            If the number of tokens is exceeded, raise a RunOnSentence
            error. If set to None, keep searching until a terminal
            punctuation mark is found. This could blow up the stack.

        Returns
        -------
        str
            Returns the generated sentence.
        """
        unbounded_sentence_tokens = take_until_inclusive(
                partial(re.match, TERMINAL_PUNCTUATION),
                self.generate_tokens(start=start))
        sentence_tokens = islice(unbounded_sentence_tokens, max_tokens)
        untrimmed_output = concatenate_grams(sentence_tokens)
        terminal_regex = fr'(?<={TERMINAL_PUNCTUATION}).*$'
        match = re.search(terminal_regex, untrimmed_output)
        if match:
            return untrimmed_output[:match.span()[0]]
        else:
            raise RunOnSentenceError

    def generate_text(self,
                      start: str = '',
                      n_tokens: int = 30) -> str:
        """
        Generates some text by traversing the Markov chain.

        Parameters
        ----------
        start : str, default=''
            Text that the generated text should start with. If empty,
            the generated text will start at a random node of the
            Markov chain.

        n_tokens : int, default=30
            Number of tokens in the generated text.

        Returns
        -------
        str
            Returns the generated text.
        """
        generated_tokens = islice(
                self.generate_tokens(start=start),
                n_tokens)
        return concatenate_grams(generated_tokens)


def tokenize(n_gram: int, unigram_regex: str, text: str) -> List[str]:
    r"""
    Tokenize a given text.

    Parameters
    ----------
    n_gram : int
        The number of unigrams that a token consists of.

    unigram_regex : int
        Regular expression encoding what constitutes a unigram.

    text : str
        The text to tokenize.

    Returns
    -------
    List of strings
        List of tokens in the text.

    >>> tokenize(2, r'\b\w+\b|,', 'please, tokenize me')
    ['please,', tokenize', 'tokenize me']
    """
    unigrams = re.findall(unigram_regex, text)
    return [concatenate_grams(tuple_gram)
            for tuple_gram in zip(*(unigrams[i:] for i in range(n_gram)))]


T = TypeVar('T')


def take_until_inclusive(predicate: Callable[T, bool],
                         iterator: Iterator[T]) -> Iterator[T]:
    """
    Return an iterator up to and including when a given predicate fails.
    Similar to itertools.takewhile.

    Example
    -------
    >>> x = [1, 2, 3, 4, 5, 6, 7, 4]
    >>> it = take_until_inclusive(lambda x: x + 1 == 5, x)
    >>> [y for y in it]
    [1, 2, 3, 4]
    """
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


class NotFittedError(Exception):
    pass


class RunOnSentenceError(Exception):
    pass
