var a = {
	'Arhar/Tur': ['Autumn','Kharif','Rabi','Summer','Whole Year','Winter'],
    'Gram': ['Autumn','Kharif','Rabi','Summer','Whole Year','Winter'],
    'Jowar': ['Autumn','Kharif','Rabi','Summer','Whole Year','Winter'],
    'Maize': ['Autumn','Kharif','Rabi','Summer','Whole Year','Winter'],
    'Moong(Green Gram)': ['Autumn','Kharif','Rabi','Summer','Whole Year','Winter'],
    'Rice': ['Autumn','Kharif','Rabi','Summer','Whole Year','Winter'],
    'Sugarcane': ['Autumn','Kharif','Rabi','Summer','Whole Year','Winter'],
    'Urad': ['Autumn','Kharif','Rabi','Summer','Whole Year','Winter'],
    'Wheat': ['Autumn','Kharif','Rabi','Summer','Whole Year','Winter'],
    'Groundnut': ['Autumn','Kharif','Rabi','Summer','Whole Year','Winter'],
    'Sesamum': ['Autumn','Kharif','Rabi','Summer','Whole Year','Winter'],
    'Soyabean': ['Autumn','Kharif','Rabi','Summer','Whole Year','Winter'],
    'Onion': ['Autumn','Kharif','Rabi','Summer','Whole Year','Winter'],
    'Potato': ['Autumn','Kharif','Rabi','Summer','Whole Year','Winter'],
    'Dry chillies': ['Autumn','Kharif','Rabi','Summer','Whole Year','Winter'],
    'Rapeseed &Mustard': ['Autumn','Kharif','Rabi','Summer','Whole Year','Winter']
}


function selCrop() {
	var crop = document.getElementById("cr").value;
    console.log(crop);

    str = "";
    a[crop].forEach(function(c) {
        str = str + "<option>" + c + "</option>";
    });
    document.getElementById("se").innerHTML = str;
}


if(window.location.pathname == "/") {
    window.onload = selCrop();
}