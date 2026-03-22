def select_best_model(results):

    best_model = None
    best_score = 0

    for name, (model, score) in results.items():

        if score > best_score:

            best_model = name
            best_score = score

    return best_model,best_score