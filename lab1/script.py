import re
from collections import Counter
import math
import numpy as np
#task 1
class NGramModel:
    def __init__(self, n):
        self.n = n  
        self.ngram_counts = Counter()  
        self.context_counts = Counter()  # Counts of (n-1)-grams, which are the contexts

    def tokenize(self, text):
        """Tokenize the input text into a list of words."""
        return ['<s>'] + re.findall(r'\w+|\S', text.lower()) + ['</s>']

    def update_counts(self, tokens):
        """Update counts of n-grams and (n-1)-grams from the list of tokens."""
        for i in range(len(tokens) - self.n + 1):
            # Get the n-gram and (n-1)-gram 
            ngram = tuple(tokens[i:i+self.n])
            context = tuple(tokens[i:i+self.n-1])
            # Update the counts
            self.ngram_counts[ngram] += 1
            self.context_counts[context] += 1

    def train(self, text):
        """Train the n-gram model on the given text."""
        tokens = self.tokenize(text)
        self.update_counts(tokens)

    def calculate_probability(self, ngram):
        """Calculate the MLE probability of an n-gram."""
        context = ngram[:-1]
        if self.context_counts[context] == 0:
            # Handling the case of a context with zero count
            return 0
        return self.ngram_counts[ngram] / self.context_counts[context]

    def get_ngram_probability(self):
        """Return a dictionary of n-grams and their MLE probabilities."""
        probabilities = {}
        for ngram in self.ngram_counts:
            probabilities[ngram] = self.calculate_probability(ngram)
        return probabilities


# NGramModel for trigrams (n=3)
ngram_model = NGramModel(n=3)

# training data --> tokenize it --> update the n-gram and context counts
with open('wiki.train.tokens', 'r', encoding='utf-8') as file:
    data = file.read()

ngram_model.train(data)

ngram_probabilities = ngram_model.get_ngram_probability()

# Output the total number of trigrams and the first few trigram probabilities to check
total_trigrams = len(ngram_probabilities)
sample_trigrams = list(ngram_probabilities.items())[:5]


print("Total number of trigrams: " + str(total_trigrams))
print("Sample trigram probabilities:" + str(sample_trigrams)) 

#######################################################################
#task 2

class InterpolatedNGramModel(NGramModel):
    def __init__(self, max_n):
        # n-gram models for all n from 1 to max_n
        self.models = [NGramModel(n) for n in range(1, max_n + 1)]
        self.max_n = max_n
        # lambda weights for interpolation (uniform distribution)
        self.lambdas = [1 / max_n] * max_n 

    def train(self, text):
        # Train each n-gram model
        for model in self.models:
            model.train(text)

    def calculate_interpolated_probability(self, ngram):
        """Calculate the interpolated probability of the n-gram."""
        interpolated_prob = 0 # Interpolated probability to be returned
        for i, model in enumerate(self.models):
            # Get the sub-ngram for the model of order i+1
            sub_ngram = ngram[-(i+1):]
            # weighted probability
            interpolated_prob += self.lambdas[i] * model.calculate_probability(sub_ngram)
        return interpolated_prob

def calculate_perplexity(model, text):
    """Calculate the perplexity of the model given the text."""
    tokens = model.tokenize(text)
    N = len(tokens) - model.max_n + 1  # Number of n-grams
    log_probability_sum = 0

    for i in range(model.max_n - 1, len(tokens)):
        ngram = tuple(tokens[i - model.max_n + 1:i + 1])
        probability = model.calculate_interpolated_probability(ngram)
        if probability > 0:
            log_probability_sum += math.log(probability)
        else:
            # Handling the case of zero probability
            log_probability_sum += math.log(1e-10)  # Small constant to avoid log(0) error

    # Calculate perplexity
    perplexity = math.exp(-log_probability_sum / N)
    return perplexity

# Load the test data
with open('wiki.test.tokens', 'r', encoding='utf-8') as file:
    test_data = file.read()


interp_ngram_model = InterpolatedNGramModel(max_n=3)
interp_ngram_model.train(data)	
# Calculate the perplexity of the interpolated model on the test data
test_perplexity = calculate_perplexity(interp_ngram_model, test_data)

print("Interpolated test perplexity: " + str(test_perplexity))


#################################################################
#task 3

class EMInterpolatedNGramModel(InterpolatedNGramModel):
    def optimize_lambdas(self, validation_text, tolerance=1e-4, max_iterations=100):
        """Optimize the lambda values using the EM algorithm."""
        tokens = self.tokenize(validation_text)
        N = len(tokens) - self.max_n + 1  # Number of n-grams

        # Initialize lambda weights (if not initialized)
        if not hasattr(self, 'lambdas') or len(self.lambdas) != self.max_n:
            self.lambdas = np.array([1 / self.max_n] * self.max_n)

        for iteration in range(max_iterations):
            lambda_updates = np.zeros(self.max_n)
            old_lambdas = self.lambdas.copy()

            # E-step: Calculate expected counts
            for i in range(self.max_n - 1, len(tokens)):
                ngram = tuple(tokens[i - self.max_n + 1:i + 1])
                total_prob = sum(self.lambdas[j] * self.models[j].calculate_probability(ngram[-(j+1):])
                                 for j in range(self.max_n))
                if total_prob > 0:
                    for j in range(self.max_n):
                        prob = self.models[j].calculate_probability(ngram[-(j+1):])
                        lambda_updates[j] += (self.lambdas[j] * prob) / total_prob

            # M-step: Update lambda values
            self.lambdas = lambda_updates / np.sum(lambda_updates)

            # Check for convergence
            if np.sum(np.abs(self.lambdas - old_lambdas)) < tolerance:
                print(f"Converged at iteration {iteration}")
                break
            
            # Print the lambda values and change in lambdas for this iteration
            print(f"Iteration {iteration}: Lambda values = {self.lambdas}, Change = {np.sum(np.abs(self.lambdas - old_lambdas))}")


em_interp_ngram_model = EMInterpolatedNGramModel(max_n=3)
em_interp_ngram_model.train(data)
# Load the validation data
with open('wiki.valid.tokens', 'r', encoding='utf-8') as file:
    validation_data = file.read()

# Optimize the lambda values using the EM algorithm on the validation data
em_interp_ngram_model.optimize_lambdas(validation_data)
full_em_test_perplexity = calculate_perplexity(em_interp_ngram_model, test_data)

print("EM lambda values: " + str(em_interp_ngram_model.lambdas))
print("EM test perplexity: " + str(full_em_test_perplexity))
