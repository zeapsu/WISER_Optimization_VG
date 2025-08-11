num_vars=31
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
    'adaptive': {
        'local_search_doe': 'adaptive',
        'local_search_num_bitflips': 1,
        'local_search_maxiter': None,
        'local_search_maxepoch': 1000,
        'local_search_maxfevals': min(2**12, 20 * num_vars)}
    }