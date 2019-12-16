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


