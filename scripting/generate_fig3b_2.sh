#!/bin/bash

figname=fig3b_2

# Location of Adobe Illustrator
ILLUSTRATOR_PATH="/Applications/Adobe Illustrator 2023/Adobe Illustrator.app"

# Base project directory, using tilde without expansion as explained below.
PROJ_DIR_TILDE="~/Documents/Projects/mescs-invitro-facs"
PROJ_DIR="${PROJ_DIR_TILDE/#\~/$HOME}"

# Template .ai file, and the folder containing images linked in the template.
# Note that the folder name will be replaced, and therefore needs to use the
# tilde explicitly, without substitution, for the filename.
template_fpath=$PROJ_DIR/scripting/templates/template_${figname}.ai
template_linkdir=$PROJ_DIR_TILDE/scripting/placeholders/images_${figname}

# This is where all generated ai files will be stored, one for every run below.
aioutdir=${PROJ_DIR}/out/3b_isolate2/ai
mkdir -p $aioutdir

# Script to modify the links in an .ai file, with placeholder in/out files,
# and temporary generated script, with placeholder in/out files replaced
scriptfpath=$PROJ_DIR/scripting/modify_links.jsx
tmp_script_fpath=$PROJ_DIR/scripting/_tmp_modify_links.jsx

# Directories containing images corresponding to trained models.
rundirs=(
    out/3b_isolate2/images/gexp3d/2.0
    out/3b_isolate2/images/gexp3d/2.5
    out/3b_isolate2/images/gexp3d/3.0
    out/3b_isolate2/images/gexp3d/3.5
    out/3b_isolate2/images/gexp3d/4.0
    out/3b_isolate2/images/gexp3d/4.5
    out/3b_isolate2/images/gexp3d/5.0
)

# Main Loop
for rd in ${rundirs[@]}; do
    tp=$(basename $rd)
    echo $tp
    fname=${figname}_${tp}
    cp $template_fpath $aioutdir/$fname.ai
    open -a "$ILLUSTRATOR_PATH" $aioutdir/$fname.ai
    sed -e "s|<OLD_FOLDER_PATH>|$template_linkdir|" \
        -e "s|<NEW_FOLDER_PATH>|$PROJ_DIR/$rd|" $scriptfpath > $tmp_script_fpath
    osascript -e 'tell application "Adobe Illustrator" to do javascript file "'"$tmp_script_fpath"'"';
    rm $tmp_script_fpath
    rm $aioutdir/$fname.ai  # remove the ai file, keeping only the pdf
done
echo Done!