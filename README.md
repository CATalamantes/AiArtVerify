# YouTube Trending Video Predictor

**Predicting Virality Through Machine Learning**

Prabesh Aryal · Rahma Ibrahim · Piyush Singh · Cristobal Talamantes · Gloria Abolade

---

## What Are We Investigating?

- What factors drive YouTube videos to trend and achieve massive visibility on the platform?
- Can AI accurately predict a video's trending potential using only its metadata, before it's published?
- Can we build an A/B testing simulator to compare YouTube video titles and optimize content before posting?

## Why YouTube Virality?

**Billions of Daily Users** — YouTube's trending page shapes what content reaches massive audiences, making it one of the most influential discovery engines online.

**Real-World Impact** — Creators, educators, journalists, and marketers all depend on platform reach. Understanding virality bridges data science with media and communication.

**Video Is Dominant** — Video is the dominant form of online communication. Predicting engagement has never been more relevant — or more valuable.

## Core Research Question

> *"How much of a video's popularity can be explained by its metadata alone?"*

We approach this question through two complementary framings:

| Framing | Goal |
|---|---|
| **Regression** | Predict exact view counts — a continuous target variable using metadata features |
| **Classification** | Categorize videos as low / medium / high popularity — a discrete, interpretable output |

## Dataset

### Data Sources

- [YouTube Trending Video Statistics](https://www.kaggle.com/datasets/datasnaek/youtube-new) — Kaggle (`datasnaek/youtube-new`)
- [YouTube Trending Videos – Daily Update](https://www.kaggle.com/datasets/canerkonuk/youtube-trending-videos-global/data) — Kaggle (`canerkonuk`)

### Global Coverage

US · UK · Germany · Canada · France · Russia · Mexico · South Korea · Japan · India

### Key Features

- View count, likes, dislikes, comments
- Category, tags, channel title
- Publish time and engagement metrics

## How Data Will Be Used

- Analyze YouTube trending data: views, engagement, categories, tags, publish time
- Train ML models to find key drivers of trending performance
- Help creators optimize content and publishing strategies
- Inform marketers, researchers, and recommendation systems
- Show how ML uncovers trends and supports data-driven decisions

## Models

### Baseline — Linear Regression

- Predicts view count from metadata (publish time, category, tags, likes, comments)
- Shows feature–popularity relationships
- Serves as an interpretable benchmark for advanced models

### Advanced — Random Forest

- Identifies complex, nonlinear patterns between video metadata and YouTube popularity
- Captures interactions among variables like views, likes, comments, categories, tags, and publish time
- Feature importance scores reveal which factors most heavily influence trending performance

## Recognizing & Addressing Bias

### Sources of Bias

1. **Sampling Bias** — Datasets include only trending videos, over-representing successful content.
2. **Geographic Bias** — Some regions have more data; models may not generalize equally across countries.
3. **Historical Bias** — Past data may not reflect current platform algorithms or creator trends.

### Mitigation Strategies

1. **Showcase Results** — Explain to the user why the model reached its conclusion.
2. **Break It Down** — Analyze results separately across categories, regions, and platforms.
3. **Be Transparent** — Clearly communicate that the model identifies patterns but cannot guarantee whether a post will go viral.

## Project Structure

```
HitOrFlop/
├── data/                  # Raw YouTube trending video datasets
├── cleaning.py            # Data cleaning and preprocessing
├── eda.py                 # Exploratory data analysis
├── featureExtraction.py   # Feature engineering from cleaned data
└── README.md
```

## Getting Started

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/canerkonuk/youtube-trending-videos-global/data) and place it in the `data/` directory.
2. Run the data pipeline in order:

```bash
python cleaning.py
python eda.py
python featureExtraction.py
```

## References

1. Canerkonuk. (n.d.). *YouTube Trending Videos Global* [Data set]. Kaggle. https://www.kaggle.com/datasets/canerkonuk/youtube-trending-videos-global/data
2. Sprout Social. (n.d.). *YouTube Trends*. https://sproutsocial.com/insights/youtube-trends/
3. Chen, Y.-L., & Chang, C.-L. (2019). Early prediction of the future popularity of uploaded videos. *Expert Systems with Applications*, 133, 59–74. https://doi.org/10.1016/j.eswa.2019.05.015
4. Aljamea, A., & Zeng, X.-J. (2024). Predicting the Popularity of YouTube Videos: A Data-Driven Approach. In S. Prajapat, P. Grace, L. Yang, P. Jenkins, & N. Naik (Eds.), *Advances in Computational Intelligence Systems* (Vol. 1453, pp. 625–639). Springer International Publishing AG. https://doi.org/10.1007/978-3-031-47508-5_48
5. Stokowiec, W., Trzcinski, T., Wolk, K., Marasek, K., & Rokita, P. (2017). Shallow reading with Deep Learning: Predicting popularity of online content using only its title. arXiv. https://doi.org/10.48550/arxiv.1707.06806
