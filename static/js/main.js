videojs('intro-video', {
  playbackRates: [0.5, 1, 1.5, 2]
});


var options = {
  slidesToScroll: 1,
  slidesToShow: 3,
  loop: true,
  infinite: true,
  autoplay: false,
  autoplaySpeed: 3000,
}

// Initialize all elements with carousel class.
const carousels = bulmaCarousel.attach('.carousel', options);
