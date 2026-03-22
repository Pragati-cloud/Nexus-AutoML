def generate_report(df, problem_type, results, best_model, best_score):

    report = []

    report.append("AutoML Report")
    report.append("====================")

    report.append("\nDataset Information")
    report.append(f"Rows: {df.shape[0]}")
    report.append(f"Columns: {df.shape[1]}")
    report.append(f"Missing Values: {df.isnull().sum().sum()}")

    report.append("\nProblem Type")
    report.append(problem_type)

    report.append("\nModel Performance")

    for name, (model, score) in results.items():
        report.append(f"{name}: {score}")

    report.append("\nBest Model")
    report.append(best_model)
    report.append(f"Score: {best_score}")

    report_text = "\n".join(report)

    with open("automl_report.txt", "w") as f:
        f.write(report_text)

    return report_text