# NAVIGATOR

An AI-powered shell helper that accepts conversational requests on the command line, and translates them in bash commands.

Navigator works in different modalities: aiprompt and command: in aiprompt modality, you can describe the command to be executed informally; it will be translated in a command to be confirmed (or corrected) in the command modality. 

There is a third modality, the comment, where you can insert the comment to be associated with a command you inserted without asking to the aiprompt first. This is useful because every command inserted directly or corrected is inserted in a training file that the programmer (me) can eventually use to better train the AI model.

At the moment I decided to not include the instruments to do the training, but this can change in the future, provided I have time.

Nothing is executed without your explicit confirmation.

Note that this software is still considered in alpha stage. It is generally safe to execute, but errors/bugs can still be present.

You will need packages in the requirements.txt files and an installation of ollama. 

Please have a look to the configuration file navirc.


Due to github limitations, the model.pth file is stored on google drive at the following link:
https://drive.google.com/file/d/1zxqJQgWJNvfewB5VIO2fFcKCpjqhQF9f/view?usp=sharing


COMPLIANCE, NAVIGATOR!

Dario
