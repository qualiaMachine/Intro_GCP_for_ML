---
title: UW-Madison Cloud Resources
---

This page collects UW-Madison-specific cloud computing resources, contacts, and funding opportunities relevant to ML researchers. It is meant as a companion to the workshop material and a starting point for learners who want to continue using cloud resources after the workshop.

Much of this information is drawn from the [ML+X Nexus UW Cloud Services page](https://uw-madison-datascience.github.io/ML-X-Nexus/Toolbox/Compute/UW-Cloud-Services.html) — check there for the most up-to-date version.

## Cloud platforms at UW-Madison

UW-Madison has institutional contracts with three public cloud vendors:

- **Amazon Web Services (AWS)** — [Service page](https://it.wisc.edu/services/amazon-web-services/) | [Pricing & billing FAQ](https://kb.wisc.edu/data/page.php?id=65532)
- **Google Cloud Platform (GCP)** — [Service page](https://it.wisc.edu/services/google-cloud-platform/) | [Pricing](https://kb.wisc.edu/100173) | [Requesting a project](https://kb.wisc.edu/data/100171)
- **Microsoft Azure** — [Service page](https://it.wisc.edu/services/microsoft-azure/) | [Pricing](https://kb.wisc.edu/69212)

These services are managed by the [UW Public Cloud Team](https://kb.wisc.edu/page.php?id=109785), a cross-disciplinary group of operations, cybersecurity, and research cyberinfrastructure (RCI) professionals.

## Why use a UW-provisioned account?

A self-provisioned cloud account (one you create directly with Google or AWS) is a personal agreement between you and the vendor — it is **not** covered by UW-Madison's institutional contracts. Going through the UW Public Cloud Team gives you:

- **Negotiated pricing** via [Internet2 NET+](https://internet2.edu/cloud/cloud-solutions-community/net-plus/) agreements. GCP accounts include a [network egress waiver](https://kb.wisc.edu/100173) (up to 15% of your total bill); Azure accounts receive ~3.5% off retail pricing.
- **Lower overhead on grants** — Cloud expenses normally carry 55.5% F&A overhead. With a UW cloud account that drops to **26%**, saving ~$2,950 per $10,000 spent. See the [Cloud Computing Pilot](https://rsp.wisc.edu/proposalprep/cloudComputeInfo.cfm).
- **NIH STRIDES discounts** — Additional pricing reductions for NIH-funded researchers, layered on top of UW rates. See [STRIDES at UW-Madison](https://kb.wisc.edu/109813).
- **Security and compliance** — Accounts come with baseline [CIS benchmark](https://www.cisecurity.org/cis-benchmarks) configuration, NetID authentication, Security Command Center monitoring, and a Business Associates Agreement (BAA) for HIPAA-regulated data.
- **Dedicated support** — Email [cloud-services@cio.wisc.edu](mailto:cloud-services@cio.wisc.edu), attend [office hours](https://kb.wisc.edu/101516), or schedule a consultation.

## How to request a UW cloud account

1. **Get a DoIT Billing Customer ID** to tie cloud usage to a funding source.
2. **Fill out the [UW-Madison Cloud Account Request Form](https://kb.wisc.edu/sbsedirbs/page.php?id=104090)** — covers AWS, GCP, and Azure.
3. **For sensitive/restricted data** — complete a [Cybersecurity risk assessment](https://kb.wisc.edu/115296) before processing HIPAA, FERPA, or other regulated data.

## Research funding and credits

### Reduced F&A on grants (Cloud Computing Pilot)

The [Cloud Computing Pilot](https://rsp.wisc.edu/proposalprep/cloudComputeInfo.cfm) reduces overhead from 55.5% to 26% on cloud expenses when using a UW-provisioned account. This applies to new proposals and awards. Costs paid via purchasing card or personal accounts are charged the full rate. RSP provides [budget templates](https://rsp.wisc.edu/proposalprep/cloudComputeInfo.cfm) for proposals.

### NIH STRIDES Initiative

NIH-funded researchers get additional cloud discounts through the [STRIDES Initiative](https://kb.wisc.edu/109813). The UW cloud team can transition accounts in or out of STRIDES at any time with no data migration.

### Google Cloud Research Credits

Google offers up to **$5,000 in cloud credits** for faculty, postdoctoral, and non-profit researchers (up to $1,000 for PhD students).

- [Apply for Google Cloud Research Credits](https://edu.google.com/intl/ALL_us/programs/credits/research/)
- Applications accepted on a rolling basis; decisions typically take 6–8 weeks.

### Google Cloud Skills Boost

UW-Madison has a limited number of seats for [Google Cloud Skills Boost](https://www.cloudskillsboost.google/). Contact the Public Cloud Team at [cloud-services@cio.wisc.edu](mailto:cloud-services@cio.wisc.edu) to request access.

## Data protection and compliance

Cloud eligibility depends on your data classification:

| Data type | Cloud eligible? | Requirements |
|-----------|----------------|--------------|
| Public / Internal | Yes | Standard UW cloud account |
| Sensitive | Yes, with assessment | [Cybersecurity risk assessment](https://kb.wisc.edu/115296) required |
| Restricted (HIPAA, etc.) | Yes, with assessment | Risk assessment + risk executive approval + HIPAA-eligible services |

Key compliance resources:

- [Data classification policy](https://kb.wisc.edu/itpolicy/page.php?id=59205)
- [Data elements allowed in public cloud](https://kb.wisc.edu/100124)
- [GCP for sensitive and restricted data](https://kb.wisc.edu/115296)
- [Shared responsibility model](https://kb.wisc.edu/data/page.php?id=115300)
- [HIPAA Security Program](https://it.wisc.edu/about/division-of-information-technology/enterprise-information-security-services/office-of-cybersecurity/hipaa-security-program/)
- SMPH researchers using Azure: contact [platformx-support@mailplus.wisc.edu](mailto:platformx-support@mailplus.wisc.edu) about [Platform X](https://it.wisc.edu/services/microsoft-azure/) for HIPAA workloads.

## On-campus compute alternatives

Cloud is not the only option. UW-Madison offers several on-campus resources that are **free for UW researchers**:

### Center for High Throughput Computing (CHTC)

[CHTC](https://chtc.cs.wisc.edu/) is UW-Madison's core research computing center, providing access to 20,000+ CPU cores and hundreds of GPUs (including A100s) at no cost to UW researchers. Key features:

- **GPU Lab** — Supports up to dozens of concurrent GPU jobs per user, including 40 GB and 80 GB A100s, with runtimes from hours to seven days.
- **Research facilitation** — Personalized consultations, online guides, and drop-in office hours to help you get started.
- **HTCondor** — CHTC's job scheduler lets you submit large batches of independent training runs (e.g., hyperparameter sweeps) across many machines.

CHTC is a strong choice for researchers who need GPU access but do not need cloud-specific services like managed APIs or cloud storage.

For more details, see the [CHTC page on Nexus](https://uw-madison-datascience.github.io/ML-X-Nexus/Toolbox/Compute/CHTC.html).

### BadgerCompute

[BadgerCompute](https://uw-madison-datascience.github.io/ML-X-Nexus/Toolbox/Compute/BadgerCompute.html) is a lightweight, NetID-authenticated Jupyter notebook service available to UW-Madison users. It is suitable for quick prototyping and small-scale work without spinning up cloud resources.

### Google Colab

[Google Colab](https://uw-madison-datascience.github.io/ML-X-Nexus/Toolbox/Compute/GoogleColab.html) provides free cloud-based Jupyter notebooks with optional GPU access. It is not a UW service, but it is a useful option for quick experiments and teaching.

## Getting help

- **Public Cloud Office Hours** — Thursdays, 2:00–3:15 PM via [Zoom](https://kb.wisc.edu/101516). Open to the entire UW community.
- **Cloud Community** — Join the [UW Cloud Community](https://it.wisc.edu/research-ci/building-cloud-community-at-uw-madison/) group, which meets every other month to share cloud computing experiences.
- **Email** — [cloud-services@cio.wisc.edu](mailto:cloud-services@cio.wisc.edu)
- **KnowledgeBase** — [kb.wisc.edu](https://kb.wisc.edu/page.php?id=109785) for FAQs, pricing details, and how-to guides.
- **ML+X Community** — Join [ML+X](https://hub.datascience.wisc.edu/communities/mlx/) for monthly meetings on machine learning and AI at UW-Madison. Contact [endemann@wisc.edu](mailto:endemann@wisc.edu) or join the `#ml-community` channel in the [Data Science Hub Slack](https://hub.datascience.wisc.edu/).
- **RCI** — The [Research Cyberinfrastructure](https://it.wisc.edu/about/division-of-information-technology/research-cyberinfrastructure/) team can help with architecture design, cost estimates, and comparing cloud vs. on-premises options. Email [rci@g-groups.wisc.edu](mailto:rci@g-groups.wisc.edu).

## Related resources

- [Intro to GCP for ML & AI](https://uw-madison-datascience.github.io/ML-X-Nexus/Learn/Workshops/Intro-GCP.html) — This workshop on Nexus.
- [Intro to AWS SageMaker for Predictive ML/AI](https://uw-madison-datascience.github.io/ML-X-Nexus/Learn/Workshops/Intro-Amazon_SageMaker.html) — Companion workshop for AWS.
- [UW Generative AI Services & Policies](https://uw-madison-datascience.github.io/ML-X-Nexus/Toolbox/GenAI/GenAI-at-UW-Madison.html) — UW-vetted AI tools including pay-as-you-go cloud AI services.
- [Introduction to AWS for Researchers (RCI)](https://researchci.it.wisc.edu/introduction-to-aws-for-researchers/) — RCI's guide for getting started with AWS.
