
Install in development mode. 


    conda develop ../conda-build



Running the interlinks filter
First, build the reference for your own site, which includes an objects.json inventory:

    python -m quartodoc build

Second, retrieve the inventory files for any other sources:

    python -m quartodoc interlinks

Finally you should see the filter run when previewing your docs:

    quarto preview


To render documentation as markdown:


    quartodoc build


Uninstall 

    conda develop ../conda-build -u
