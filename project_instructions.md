# Welcome to project Okavango
---

## Scenario

Your group is participating in a two-day hackathon where the goal is developing a light-weight data analysis tool that will be used in environmental protection using the **MOST RECENT DATA AVAILABLE**.

## Goal

For this project, we will be using data from [Our World in Data](https://www.ourworldindata.org). The datasets can be found in:

1. [Anual Change in forest area](https://ourworldindata.org/deforestation)
1. [Annual deforestation](https://ourworldindata.org/deforestation)
1. [Share of land that is protected](https://ourworldindata.org/sdgs/life-on-land)
1. [Share of land that is degraded](https://ourworldindata.org/sdgs/life-on-land)
1. [A fifth dataset you find relevant](https://ourworldindata.org/sdgs/life-on-land)
1. [Map dataset: Admin 0 – Countries; Download Countries](https://www.naturalearthdata.com/downloads/110m-cultural-vectors/)

A little help for the latter file:
```bash
import geopandas as gpd
world = gpd.read_file("ne_110m_admin_0_countries.zip")
```

Go over the datasets with your group. Check the info on the website before you start. 

<div class="alert alert-danger">
    <b> THE MOST IMPORTANT TOOLS FOR A DATA SCIENTIST ARE PATIENCE AND COMMUNICATION</b>
    <br>
    <b> Discuss the contents of the datasets with your colleagues. Understanding the data is a priority. </b>
</div>

Use whatever python tools you find apropriate.

## Structure of the project

You are going to build a **[Streamlit app](https://streamlit.io/)** that will showcase your analysis.  
Since you are also using geographical data, you will be using [geopandas](https://geopandas.org).
Use a project structure like this:

```bash
|__downloads/
|__app
|    |__your .py files here
|__tests
|    |__your test files here
|__notebooks
|    |__your notebooks here
|__.gitignore
|__README.md
|__LICENSE.md
|__main.py
```

### Day 1, Phase 1

- One of you will create a github repository (it does not matter who). __THE NAME OF THE REPOSITORY MUST BE "Group_X" where X is the letter of the group! Use a capital letter!__
- Initialize the repo with a README.md file, a proper license, and a .gitignore for the programming language you will use. The README.md file __MUST__ have your emails in a way that it is possible to copy and paste it into outlook.
- The one who created the repository will then give __Maintainer__ permissions to the rest of the group. Check under "Project Information" > "Members".
- [ ] Every element of the group clones the repository to their own laptops.

### Day 1, Phase 2, the data

- [ ] Create a python function (**Function 1**, but you'll have to call it something else) that downloads all the required datasets into the `downloads` directory.
- [ ] Create another function (**Function 2**, again, call it something PEP8 compatible) that merges the map with the datasets. Use `geopandas`. Make sure the left dataframe is the `geopandas` dataframe. See if there are changes needed in order for the merges to work.
- [ ] You decide the create a Class to better handle the data for the project. All names you use __must be PEP8 compliant, like the entire project__.
- [ ] Make a test for Function 1 and one for Function 2, in the appropriate place. If one runs **pytest** in the main directory of the project, all the tests must run.

Now it is time to integrate the functions in a class. The class will have several methods, which you will __not__ develop in the master branch in git.  
Document everything!  
Make your calls compliant with __pydantic__ and __static type checking__ when appliable.

- [ ] Integrate Functions 1 and 2 in your class: During the _init_ method, both functions are execured.
- [ ] The _init_ method must also read the datasets into corresponding dataframes which become attributes for your class.

### Day 1, Phase 3, the app itself

- [ ][ ][ ] Make a Streamlit App where you import your __Class__ and plot the maps. About your streamlit app...
        * It should have a way to select which map to plot. Each map is one of the geopandas dataframes. Only one map at the time.
        * It should plot a graph below the world map. According to each dataset you have, plot the most appropriate information you find, like an histogram of the top 5 countries and bottom 5 countries with annual changes in forest area.

**If you feel lost, don't hesitate to contact me.**

<div class="alert alert-info">
    <b> REMEMBER: THE MOST RECENT DATA AVAILABLE! IT MUST NOT BE HARDCODED!</b>
</div>


<div class="alert alert-info">
    <b> REMEMBER: The first delivery is until March 2 23:59:59 and it is not graded. It is used as course correction. The delivery is the git repo link on moodle. </b>
</div>


<div class="alert alert-info">
    <b> REMEMBER: IT IS OK TO PROTOTYPE CODE IN NOTEBOOKS, BUT THE CLASS MUST BE IN A SINGLE .py FILE! </b>
    <br>
    <b> Prototyping notebooks must have their own separate directory.</b>
    <br>
    <b> We will only consider contents in your "master" repository.</b>
</div>

<div class="alert alert-warning">
    <b>When in doubt, ask.</b>
</div>

<br>
<br>

# Welcome to project Okavango - part 2

## Rules
1. Be sure that your group already submitted [the link to the repo on moodle](https://moodle.novasbe.pt/mod/assign/view.php?id=48479).
2. We will consider work done until 23:59:59, Friday 20 March 2026. Remember: timestamps are recorded!

## Scenario

Your group is participating in a two-day hackathon where the goal is developing a light-weight data analysis tool that will be used in environmental protection using the **MOST RECENT DATA AVAILABLE**.

## Additional Goals

You finished an analysis tool that will enable your group to pinpoint at-risk natural regions of the world.  

It's time to scale up the usage of the tool using a bit of AI.  

Build upon your **[Streamlit app](https://streamlit.io/)** and use whatever additional tools you find appropriate.  

You are building a **proof of concept** application. You are expected to use free software, so results may vary. Aim for good quality outcomes, but focus first that the technical part is working.

## Scenario (continuation)

Day one has come and gone.

It is now time to do the final tasks and all of the polishing. As you know, your project might be picked up for an analysis presentation, you add an introduction about your group on the _README.md_ file. Be sure to add your **names**, **your student numbers** and **your e-mails**.

Day two is beginning.

### Day 2, Phase 1: AI Workflow

Let's continue to expand the app. We are going to use AI in a workflow. The purpose is to select a pair of geographical coordinates, a value for the zoom, and obtain an image from that area. Afterwards, we're going to get a description of that image to check if it is an at-risk area. There are many ways to do this.

- [ ] Add a page 2 to your streamlit app. Unless explicitly mentioned, you will work there from now on. You should have a way to select **latitude**, **longitude**, **zoom**. Add whichever other widgets you find appropriate.
- [ ] Find a way to download an image from **ESRI World Imagery** given the geographical settings. Rememeber, you must use free alternatives. Download the image into a new **images** directory.
- [ ][ ] Use [ollama](https://ollama.com/) inside python. Select a model to look at the image and generate a description of the image (this might take a minute or two). The description the model provides must be shown alongside the image. If the model does not exist in the user's PC, the code must pull it.
- [ ][ ] Add a display window below both image and image description. Given the image description, what questions should you ask behind the scenes to the description to see if the area is in environmental danger? Ask these questions to a model in ollama and get an automatic response. Create a visual way to indicate if the models flagged the area as being at environmental risk. If the model does not exist in the user's PC, the code must pull it.

### Day 2, Phase 2: Data governance.

- [ ] Make sure you have a **models.yaml** file in your main directory. This file should contain:
1. The model you are using to analyse the image.
1. The prompt you are using for the model to analyse the image.
1. Other image settings you deem fit.
1. The model you are using to analyse the deescription.
1. The prompt you are using in this model
1. Other settings you deem fit for this model.
This file should be read by your app to configure at AI workflow.

- [ ] Add a **database** directory to the project. In this directory, create a **images.csv** file. This file should be a table, something like this (example values):

| timestamp | latitude | longitude | zoom | image_description | image_prompt | image_model | text_description | text_prompt | text_model | danger |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| timestamp of the run | -3 | -3 | 17 | "Some nice flowers" | "what is in the image?" | quak:2b | "Seems ok" | "Is there an environmental danger in the description?" | duck:1b | "N" |

You may have more or less columns. The important part is that someone with your settings would reach a very similar result.

Every time you run the classification pipeline, a new row is appended to the csv.

- [ ] To save compute, every time you run the evaluation pipeline, first check if the image for the current settings laready exists in the database. If so, don't run the pipeline and just display what already exists in memory. Yes, this means you must find a way to store the images.

### Day 2, Phase 3: Cleaning up

- [ ] Make sure you have all the files in the repo. Try to download the repo and see if you can make everything run with your instructions. Add instructions in the README.md on how to install and run everything. Write a small essay at the the end of your README.md on how you think this project could help with the [UN's SDGs](https://sdgs.un.org/goals). Identify which goals you associate with the project. Showcase at least three examples of your app successfully identifying environmental dangers (images in the README with text).

---
## Grading

Between the two parts, there are 20 gradable items in both Part 1 and 2. Every [] is 1 point out of 20.

<div class="alert alert-danger">
    <b> REMEMBER: IT IS OK TO PROTOTYPE CODE IN NOTEBOOKS OR OTHER FILES </b>
    <br>
    <b> The final delivery of the project is the app. </b>
    <br>
    <b> We will only consider contents in your "main" repository before the end of the deadline.</b>
</div>