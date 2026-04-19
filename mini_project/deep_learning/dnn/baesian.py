def calculate_posterior_probability(prior_prob, likelihood_jump, likelihood_not_jump):
    ball_prob = prior_prob * likelihood_jump + (1-prior_prob) * likelihood_not_jump
    
    return round(likelihood_jump * prior_prob/ball_prob, 2)

print(calculate_posterior_probability(0.1, 0.8, 0.3))