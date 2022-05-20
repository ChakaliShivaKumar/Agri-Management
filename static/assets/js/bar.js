$(function(){
  //get the bar chart canvas
  var ctx = $("#bar-chartcanvas");

  //bar chart data
  var data = {
    labels: crop,
    datasets: [
      {
        data: pred,
        backgroundColor: "rgba(10,20,30,0.3)",
        borderColor: "rgba(10,20,30,1)",
        borderWidth: 1
      }
    ]
  };

  //options
  var options = {
    responsive: true,
    title: {
      display: true,
      position: "top",
      text: "Other Crops",
      fontSize: 18,
      fontColor: "#111"
    },
    legend: {
      display: false,
      position: "bottom",
      labels: {
        fontColor: "#333",
        fontSize: 16
      }
    },
    scales: {
      yAxes: [{
        scaleLabel: {
          display: true,
          labelString: 'Crop Yield (Tonnes)'
        }
      }],
      xAxes: [{
        scaleLabel: {
          display: false,
          labelString: 'Crop'
        }
      }]
    } 
  };

  //create Chart class object
  var chart = new Chart(ctx, {
    type: "bar",
    data: data,
    options: options
  });
});