$(function(){
  var options = {
    slidesToScroll: 1,
    slidesToShow: 3,
    loop: true,
    infinite: false,
    autoplay: false,
    autoplaySpeed: 3000,
  }
  const carousel_3d_photo = bulmaCarousel.attach('#carousel-3d-photo', options);

  var options_demo = {
    slidesToScroll: 1,
    slidesToShow: 5,
    loop: true,
    infinite: false,
    autoplay: false,
  }
  const carousel_demo = bulmaCarousel.attach('.demo-carousel', options_demo);

  $(".image-compare").twentytwenty({
    default_offset_pct:	0.7,
    before_label: '',
    after_label: '',
  });
});
