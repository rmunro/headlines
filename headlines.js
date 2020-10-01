/**
 * @fileoverview functionality to support an interface for labeling headlines 
 *
 * most of the processing happens in headlines.py, and this javascript file
 * is lightweight code to move data between the python processing and the html interface
 *
 */

var data = []; // data currently being annotated

var focus_annotation = false; // whether annotation window is in focus

var instructions = "Press 'a' to accept this headline for the label. <br /> \
      Press 'n' to indicate this does not belong to the label.<br /> \
      Press 'i' to save this headline as an interesting example:";
var click_here = "Click here to begin annotating data."


/**
* Save the predictions from the current model to file
*/
async function save_current_predictions(){
    current_label = await eel.current_label()();
    
    //NB: *not* asynchronous - this could take some time 
    eel.save_current_predictions()();

    alert("Predictions being saved to new file in 'data/"+current_label+"/'. This might take some time!"); 
}

/**
 * Creates a new label for the person to start annotating
 * Switches to that label when it already exists
 */
async function create_label(){
    
    let label = document.getElementById('label').value;

    label_div = document.getElementById('label_message');
    label_div.innerHTML = "loading...";
    
    let success = await eel.create_label(label)(); // Call to python function

    if(success){
        label_div.innerHTML = "Label: "+label;
        document.getElementById('label').style.display = "none";
        document.getElementById('label_button').style.display = "none";
        document.getElementById('prior_labels').style.display = "none";
        
        document.getElementById('filters').style.display = "block";
        document.getElementById('examples').style.display = "block";
        document.getElementById('annotation').style.display = "block";
        
        
        get_data_to_annotate();
    }
    else{
        label_div.innerHTML = "Invalid label name: only letters, numbers, space, hyphen, and underscore are permitted";        
    }
}
       
       
async function get_existing_labels(){
    let labels = await eel.get_existing_labels()(); // Call to python function
    labels_div = document.getElementById('existing_labels');
    for(var i = 0; i < labels.length; i++){
        label = labels[i]
        var existing_label = document.createElement("div");   
        existing_label.setAttribute("class", "existing_label");
        existing_label.setAttribute("onClick", "choose_existing('"+label+"');");
        existing_label.innerHTML = label;
        labels_div.appendChild(existing_label);
    }
}       

function choose_existing(label){
    document.getElementById('label').value = label;
    create_label();
}
   
/**
 * Records annotation and pass to python function to store
 */   
async function add_annotation(positive=true){
    fields = data[0] // only annotating one at a time for now
    console.log(fields)

    let message = await eel.add_annotation(fields, positive)(); // Call to python function
    get_data_to_annotate();
}       


async function tag_as_interesting(){
    fields = data[0] // only annotating one at a time for now
    let message = await eel.tag_as_interesting(fields)(); // Call to python function
    alert("this headline has been saved as 'interesting' ");
}       


/**
 * Get examples of headlines for the label for each year
 * Display these in the interface
 */
async function get_positives_per_year(){
    let samples = await eel.get_positives_per_year()(); 
    console.log(samples);
    var count = 0;
    for (var year in samples) {
        console.log(year);
        headlines = samples[year];
        div_id = 'year_'+year;
        year_div = document.getElementById(div_id);
        year_div.innerHTML = '<h4>'+year+'</h4>';
        console.log(headlines);
        
        for(var headline_key in headlines){
            headline_details = headlines[headline_key];
            console.log(headline_details);
            headline = headline_details[1];
            url = headline_details[2];
            // year_div.innerHTML += '<a href="'+url+'" target="_blank">'+headline+'</a><br />';
            
            var link = document.createElement("a");   
            link.setAttribute('href', url);
            link.setAttribute('target', '_blank');
            link.setAttribute('style', 'display:block; clear:left;');
            link.innerText = headline;
            year_div.append(link);   
            count++; 
        }
    }
    
    if(count > 0){
        document.getElementById('examples').style.display = "block";
    }
}


/**
 * Request unlabeled item to annotate
 */
async function get_data_to_annotate(number=1){
    filter = document.getElementById("filter").value;
    console.log(filter)
    year = document.getElementById("year").value;
    uncertainty = document.getElementById("uncertainty").checked;
    console.log(year)
    
    data = await eel.get_data_to_annotate(number, filter, year, uncertainty)(); // Call to python function
    
    headline_div = document.getElementById("headline");
    
    if(data[0] === undefined || data[0][1] === undefined){
        headline_div.innerHTML = "Error: no headlines found meeting the filter conditions"; // there were less than 10 results
    }
    else{
        headline_div.innerHTML = data[0][1];
    }
    

}


/**
 * Monitor for keyboard short-cut annotations 
 */
 document.addEventListener("keypress", function onEvent(event) {
    if(focus_annotation){
        if (event.key === "a") {
            add_annotation(true);
        }
        else if (event.key === "n") {
            add_annotation(false);
        }
        else if (event.key === "i") {
            tag_as_interesting();
        }
    }
    
});

/**
 * Monitor for whether the annotation window is in focus
 */
document.addEventListener("click", function onEvent(event) {
    focus_annotation = false;
    for(i = 0; i < event.path.length; i++){
        if(event.path[i].id == 'annotation'){
            focus_annotation = true;
            break;
        }
    }
    
    annotation = document.getElementById("annotation");
    instructions_div = document.getElementById("instructions");
    
    if(focus_annotation){
        annotation.setAttribute("class", "annotation_focused");
        instructions_div.innerHTML = instructions;
    }
    else{
        annotation.setAttribute("class", "annotation");        
        instructions_div.innerHTML = click_here;
    }
    
});



        