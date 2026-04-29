# Observations — S&P 500 Semantic Map

**Team members:** Mohan Golleru, Rahul.

**Embedding model:** `all-MiniLM-L6-v2` (sentence-transformers, 90 MB).
**Pipeline diagnostics:** axis-1 pole separation = 0.543, axis-2 pole separation = 0.544 (both well above the 0.30 rule-of-thumb), |cos(axis-1, axis-2)| = 0.081 — the two axes are very close to orthogonal, so the plot uses both dimensions independently rather than collapsing onto a single direction.

## What separates along each axis

**Axis 1 — Industrial / physical (−)  ↔  Digital / tech (+).** This axis pulls firms apart on whether their *name* evokes physical infrastructure or digital services. The negative pole pulls in Steel Dynamics, CF Industries, Tractor Supply, Builders FirstSource and Duke Energy; the positive pole captures Salesforce, Microsoft, Datadog, Sandisk and Oracle. Notable cross-sector drift — Fidelity National Information Services, Ares Management and Public Service Enterprise Group land deep in the tech half despite being classified as Financials and Utilities — confirms the axis is reading semantic content of the *name* (the words "Information", "Service", "Enterprise") rather than the GICS label.

**Axis 2 — Enterprise / B2B (−)  ↔  Consumer-facing (+).** This axis sorts firms by who writes the cheque. The top of the plot is dominated by household grocery, beverage and quick-service brands (Kroger, Walmart, PepsiCo, McDonald's, Yum! Brands, Chipotle, Keurig Dr Pepper); the bottom collects insurance, industrial-automation, rail-freight, enterprise software and semiconductor firms (Erie Indemnity, Rockwell Automation, Trane Technologies, CSX, Oracle, Texas Instruments). The two axes are independent enough that all four quadrants are populated: Walmart and PepsiCo land industrial-consumer (upper-left), Carnival and Domino's drift into tech-consumer (upper-right), Steel Dynamics and Tractor Supply sit industrial-enterprise (lower-left), and Microsoft, Oracle and Salesforce anchor tech-enterprise (lower-right).

## Most surprising point — Public Service Enterprise Group

Public Service Enterprise Group is a New Jersey **utility** (electric and gas), yet the embedding plants it at (x = +0.27, y = −0.21) — the deepest tech-enterprise corner, side-by-side with Microsoft and Oracle. The reason is purely linguistic: the name is a near-perfect bag-of-tech-enterprise tokens — *"Service"*, *"Enterprise"*, *"Group"* — vocabulary that the sentence transformer associates with B2B software firms far more strongly than with regulated power utilities. This is exactly the failure mode the worked example warns about: a SemAxis projection captures *how the entity is discussed* (or, here, *named*), not what it does. The same effect explains Ares **Management**, Fidelity National **Information Services** and Quanta **Services** all leaking into the tech half. A real domain analyst would either rename these firms with industry tags before embedding, or would average the company name with a one-line business description to anchor the meaning.

## What a third axis would capture

A useful third axis, conceptually orthogonal to both current ones, would be **global reach** — globally-recognized megabrand (+) versus domestic specialist (−). Within each quadrant, much of the within-cluster spread is otherwise unexplained: Walmart and Kroger both land near the consumer pole but Walmart is a global megabrand while Kroger is a U.S.-only grocer; Salesforce and Datadog both anchor the tech-enterprise corner but Salesforce is a household name in IT circles while Datadog is essentially invisible to anyone outside DevOps. A third reach axis would also explain why Apple and Alphabet sit near the origin rather than at the tech-consumer extreme: their names are so strongly associated with "global brand" that the embedding gives them a large reach component which is currently projected away when we use only two axes.
