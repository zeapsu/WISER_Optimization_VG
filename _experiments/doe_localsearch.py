doe_localsearch = {
    'long': {
        'local_search_doe': 'long',
        'local_search_num_bitflips': 1,
        'local_search_maxiter': None,
        'local_search_maxepoch': 1000,
        'local_search_maxfevals': 2**15},
    'fast': {
        'local_search_doe': 'fast',
        'local_search_num_bitflips': 1,
        'local_search_maxiter': None,
        'local_search_maxepoch': 1000,
        'local_search_maxfevals_per_variable': 2},
    'unconstrained': {
        'local_search_doe': 'unconstrained',
        'local_search_num_bitflips': 1,            # flip 1 bit at a time (standard for local search)
        'local_search_maxiter': None,              # no explicit inner iteration limit
        'local_search_maxepoch': 1000,             # try 1000 outer loop passes (fallback convergence cap)
        'local_search_maxfevals_per_variable': 1}  # try ~n function calls for an n-variable problem
    }