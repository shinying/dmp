$(window).on('load', function(){
  var options = {
    slidesToScroll: 1,
    slidesToShow: 3,
    loop: true,
    infinite: false,
    autoplay: false,
    autoplaySpeed: 3000,
  }
  bulmaCarousel.attach('#carousel-3d-photo', options);

  var options_demo = {
    slidesToScroll: 1,
    slidesToShow: 5,
    loop: true,
    infinite: false,
    autoplay: false,
  }
  bulmaCarousel.attach('.demo-carousel', options_demo);

  $(".image-compare").twentytwenty({
    default_offset_pct:	0.7,
    before_label: '',
    after_label: '',
  });
});
