"""Dataset loaders: raw source files -> canonical `Record` objects.

One module per dataset. Each benchmark has its own column names, encoding,
price spelling and ground-truth format, and keeping them apart is what stops
those quirks leaking into a shared loader full of per-source branching --
the same reason `blocking/` is a package rather than one module.

Every loader here has the same contract: it returns `Record` objects with
`entity_id` populated from the dataset's ground truth, so nothing downstream
needs to know which benchmark a record came from.
"""
