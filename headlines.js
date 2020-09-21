async function create_label(){
    
    let label = document.getElementById('label').value;

    label_div = document.getElementById('label_message');
    label_div.innerHTML = "loading...";
    
    let success = await eel.create_label(label)(); // Call to python function

    if(success){
        label_div.innerHTML = "Label: "+label;
        document.getElementById('label').style.display = "none";
        document.getElementById('label_button').style.display = "none";
        
        document.getElementById('filters').style.display = "block";
    }
    else{
        label_div.innerHTML = "Invalid label name: only letters, numbers, space, hyphen, and underscore are permitted";        
    }
}
       
async function add_annotation(index, positive=true){
    fields = data[index]
    console.log(index)
    console.log(fields)

    let message = await eel.add_annotation(fields, positive)(); // Call to python function
    
}       


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


var data = []
 
async function get_data_to_annotate(number=10){
    filter = document.getElementById("filter").value;
    console.log(filter)
    year = document.getElementById("year").value;
    console.log(year)
    
    data = await eel.get_data_to_annotate(number, filter, year)(); // Call to python function
    
    var i ;
    var tabindex = 1
    for(i=0; i <= 9; i++){
        message_id = "message_".concat(i.toString());
        message_div = document.getElementById(message_id);
        
        button_pos = '<button onclick="add_annotation('+i.toString()+');" tabindex="'+(tabindex++).toString()+'" >POS</button>';
        button_neg = '<button onclick="add_annotation('+i.toString()+',false);" tabindex="'+(tabindex++).toString()+'" >NEG</button>';
        
        console.log(i);
        if (data[i] === undefined){
            message_div.innerHTML = ""; // there were less than 10 results
        }
        else{
            headline = data[i][1];
            // TODO: add link?
            message_div.innerHTML = button_pos + button_neg +" " + headline;                
        }
    }
    
    document.getElementById('annotation').style.display = "block";

}
        