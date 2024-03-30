var imgs = [8,18,20,23,27,28,30,31,32,33];
var content = "";
imgs.forEach(elem => {
  var code = `
    <div class="item">
      <div class="image-compare is-invisible" class="twentytwenty-container">
        <img src="static/images/demo/xl/${elem}.jpg" alt="" />
        <img src="static/images/demo/xl/normal/${elem}.jpg" alt="" />
      </div>
    </div>`;
  content += code;
})
document.getElementById("normal-demo").innerHTML = content;


imgs = [2,4,8,12,17,18,23,28,31,33];
content = "";
imgs.forEach(elem => {
  var code = `
    <div class="item">
      <div class="image-compare is-invisible" class="twentytwenty-container">
        <img src="static/images/demo/xl/${elem}.jpg" alt="" />
        <img src="static/images/demo/xl/depth/${elem}.jpg" alt="" />
      </div>
    </div>`;
  content += code;
})
document.getElementById("depth-demo").innerHTML = content;


imgs = [1,2,9,10,11,12,13,14];
content = "";
imgs.forEach(elem => {
  var code = `
    <div class="item">
      <div class="image-compare is-invisible" class="twentytwenty-container">
        <img src="static/images/demo/scenes/${elem}.jpg" alt="" />
        <img src="static/images/demo/scenes/seg/${elem}.jpg" alt="" />
      </div>
    </div>`;
  content += code;
})
document.getElementById("seg-demo").innerHTML = content;


imgs = [1,2,3,4,5,6,7,8];
content = "";
imgs.forEach(elem => {
  var code = `
    <div class="item">
      <div class="image-compare is-invisible" class="twentytwenty-container">
        <img src="static/images/demo/scenes/${elem}.jpg" alt="" />
        <img src="static/images/demo/scenes/ref/${elem}.jpg" alt="" />
      </div>
    </div>`;
  content += code;
})
document.getElementById("ref-demo").innerHTML = content;


content = "";
imgs.forEach(elem => {
  var code = `
    <div class="item">
      <div class="image-compare is-invisible" class="twentytwenty-container">
        <img src="static/images/demo/scenes/${elem}.jpg" alt="" />
        <img src="static/images/demo/scenes/sha/${elem}.jpg" alt="" />
      </div>
    </div>`;
  content += code;
})
document.getElementById("sha-demo").innerHTML = content;


var vids = [1,2,3,4]
content = ""
vids.forEach(elem => {
  var code = `
    <div class="item">
      <video class="lazy" id="3d-photo-${elem}" autoplay muted loop playsinline 
             data-src="static/videos/${elem}.mp4">
        <source data-src="./static/videos/${elem}.mp4" type="video/mp4">
      </video>
    </div>`;
    content += code;
})
document.getElementById("carousel-3d-photo").innerHTML = content;