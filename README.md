# MagicDraftBot

My initial attempt at a Magic: the Gathering Draft AI. 

More detailed description of interpretable results and performance coming soon.

### File Descriptions

models.py --- the two models trained to create the Draft Agent

preprocessing.py --- some data processing techniques (e.g. label drafts via Kmeans)

create_model.py --- the script for creating the model

predict_from_log.py --- script for seeing how the bot would navigate a draft given a MTGO Draft Log

bot_table_draft.py --- script for generating a draft with 8 bots at the table

Output/ --- place to store images to show results. Currently only shows 75% test accuracy.

Data/ and Saved_Models/ --- folders in .gitignore as these cannot be in this repo due to NDA.

### Example Drafts

Below are some decks and drafts that the first iteration of this bot did on MTGO. This was with a different modeling of staying open, and no bias ceiling, but a similar implementation. I will update this section as I continue to test the bot.

Draft: https://magic.flooey.org/draft/show?id=1_fn_hicUYtWQyvXSbjjhtsIy1k

![3-0](https://pbs.twimg.com/media/ELFSpb4XkAAYCrb?format=jpg&name=small)

Draft: https://magic.flooey.org/draft/show?id=9GymyrTy70YDYfHiKh2HRvNodao

![0-0](https://pbs.twimg.com/media/ELFUnzvWkAEFtGn?format=jpg&name=small)

Draft: https://magic.flooey.org/draft/show?id=xg0h8Jo41w2TcHNwwAQ-_UlxxFQ

![0-0](https://pbs.twimg.com/media/ELT3FbHW4AA3V9z?format=jpg&name=small)



