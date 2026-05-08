for i, (n1, s1) in enumerate(REGRESSORS.items()):
    for j, (n2, s2) in enumerate(REGRESSORS.items()):
        if i < j:
            b1, _ = load_beta_pair(s1)
            b2, _ = load_beta_pair(s2)
            r = np.corrcoef(b1[:20], b2[:20])[0,1]
            print(n1, n2, r)