from sklearn.mixture import GaussianMixture


def evaluate_gmm_components(X_scaled, max_components=10):
    """Đánh giá GMM bằng BIC và AIC."""

    n_samples = len(X_scaled)
    if n_samples < 3:
        raise ValueError("Cần ít nhất 3 khách hàng để đánh giá.")

    max_components = min(max_components, n_samples - 1)

    results = {
        "n_components": [],
        "bic": [],
        "aic": []
    }

    for n in range(2, max_components + 1):
        model = GaussianMixture(
            n_components=n,
            covariance_type="full",
            random_state=42,
            n_init=3,
            reg_covar=1e-5
        )

        model.fit(X_scaled)

        results["n_components"].append(n)
        results["bic"].append(model.bic(X_scaled))
        results["aic"].append(model.aic(X_scaled))

        print(
            f"GMM components={n}: "
            f"BIC={results['bic'][-1]:.2f}, "
            f"AIC={results['aic'][-1]:.2f}"
        )

    return results


def train_gmm(X_scaled, n_components=4):
    """Huấn luyện GMM và trả về mô hình cùng nhãn cụm."""

    if len(X_scaled) < n_components:
        raise ValueError(
            "Số khách hàng phải lớn hơn hoặc bằng số thành phần."
        )

    model = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=42,
        n_init=3,
        reg_covar=1e-5
    )

    model.fit(X_scaled)
    labels = model.predict(X_scaled)

    return model, labels
