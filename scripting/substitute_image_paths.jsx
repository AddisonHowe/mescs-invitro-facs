app.preferences.setBooleanPreference("ShowExternalJSXWarning", false);
app.userInteractionLevel = UserInteractionLevel.DONTDISPLAYALERTS;

// var oldFolder = "~/Documents/Projects/plnn/out/bin_dec_models/7_24/model_phi1_1a_v_kl1_20240705_024039";
// var newFolder = "~/Documents/Projects/plnn/out/bin_dec_models/7_24/model_phi1_1a_v_mmd1_20240704_134102";

var oldFolder = String(prompt("Enter old path to images"));
var newFolder = String(prompt("Enter new path to images"));
oldFolder = oldFolder.replace("/Users/addisonhowe", "~")
newFolder = newFolder.replace("/Users/addisonhowe", "~")

var doc = app.activeDocument;
var links = doc.placedItems;

var idxsToRemove = [];  // Track the indexes of the links to remove.

var numReplaced = 0;
var numRemoved = 0;
for (var i = 0; i < links.length; i++) {
    var link = links[i];
    try {
        if (link.file && link.file.fullName.indexOf(oldFolder) !== -1) {
            var newFilePath = link.file.fullName.replace(oldFolder, newFolder);
            var newFile = new File(newFilePath);
            if (newFile.exists) {
                link.file = newFile;
                numReplaced++;
            } else {
                alert("Could not find replacement for linked item:" + 
                      link.file.fullName)
                idxsToRemove.push(i)
                numRemoved++;
            }
        }
    } catch (e) {
        alert("Caught Error:" + e)
    }
}

// Remove links in reverse order
// var nlinks = links.length;
// for (var i = 0; i < idxsToRemove.length; i++) {
//     link = links[nlinks - i - 1];
//     link.remove();
// }

alert(
    "Summary" +
    "\nTotal number of links: " + links.length +
    "\nReplaced: " + numReplaced + 
    "\nFailed to replace: " + numRemoved
)

// Save ai file
doc.save();

// doc.close();