Project AI Coloring Book Guideline

Tasks
This topic was inspired by the picture books at https://supercoolstoryteller.com/supercoolscientists. The goal is to evaluate whether current AI tools (text generation and image-to-image models) can create something similar while retaining truthfulness, and whether this can be done almost entirely automatically. If successful, a picture book featuring people from the University of Marburg could be created in time for the university's 800-year celebrations in 2027.

Source:
https://en.wikipedia.org/wiki/Marie_Curie
Drawing:
[https://huggingface.co/spaces/awacke1/Image-to-Line-Drawings, no post-processing]
Description:
Marie Curie (1867–1934) was a Polish-French scientist famous for her work on radioactivity. She was the first woman to win a Nobel Prize and the only person to win it in two different sciences (Physics and Chemistry). She discovered polonium and radium, created mobile X-ray units for WWI, and founded research institutes in Paris and Warsaw. She died from radiation exposure but remains one of history's most celebrated scientists.
[generated with Mistral.AI from the first paragraph of the wiki page]

The task is to develop a software framework to generate a coloring book automatically, given a list of names. Possible subtasks:
 * Implement / adapt crawler for wikipedia.
 * Analyse different AI models for text simplification /summarization, and image to image transformation for their suitability. Come up with a good evaluation to decide "suitability".
 * Implement a tool that checks the correctness of facts in the generated text.
 * Implement a software pipeline that automatically generates single pages of a coloring book with open source models.
 * Finetune or prompt-engineer models to optimize the output. Add postprocessing (e.g., simple image filters) to optimize the images.
 
Helpful Starters:
 * https://huggingface.co/ for AI models
 * https://opencv.org/ for classical computer vision tasks
 

Procedures and Deadlines Project Work (Master project)
| Setup Meeting | Oct 29th | Project Commitment, Teams, Organisation, Clarification on Requirements and Deliverables | XAI-Lab |
|---|---|---|---|
| MS 1 | 10.-14.11. | Definition MVP, Technologies chosen, Plan for next Sprint | Students |
| MS 2 | 15.-19.Dec. | Implementation 1st Prototype, Plan for next Sprint | Students |
| MS 3 | April | Implementation 2nd Prototype, Plan for next Sprint | Students |
| Ms 4 | July | Implementation 3rd Prototype, Plan for next Sprint | Students |
| Final Deliverable | 29.9.2026 | Final Deliverable (Details below) | Students |

Final Deliverable
 * Final Prototype
 * Software incl. Documentation
 * Demo Video (3-5 minutes, screencast demonstrating the system)
 * End Report (8 pages, 2 column ACL style)
 
Report Format
We'll be going for a two-column conference style paper of maximum 8 pages. Additionally, you have unlimited space for references and informative appendices. The style files to be used are those of the Association of Computational Linguistics (ACL) conference, which can be found here, together with paper templates: https://github.com/acl-org/acl-style-files/

Report Structure and Contents
You are free to decide on the structure and contents of your paper, but we would expect to see sections or subsections dealing with the following topics.
 * Abstract. A short 100-200 words summary of your work.
 * Introduction. Which problem are you trying to solve / which research question(s) are you trying to answer and why? Brief overview of your approach and key results.
 * Related work. Discuss work by others who have tackled the same or similar problems or who have used similar methods and describe how your own work builds on this related work and what makes it different.
 * Data/resources. Which dataset did you use? If you used existing dataset(s), mention the source. If you created your own dataset (e.g., for evaluation), describe how you did this. Describe the properties of the dataset. What kind of documents does it consist of? How many documents does it contain? With which information has it been labelled (and how / by whom)? Etc. Also describe if you have done any preprocessing of the dataset.
 * Method. Describe which method you used to answer your question/solve the problem. This is the section where you describe the implementation of your system. What design decisions did you make and why? What baselines are you using to show that your system works over established methods in the field? Refer to literature where relevant.

 * Evaluation. Describe your evaluation method/metrics. Discuss significance testing as appropriate.

