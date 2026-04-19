// 스크롤하면 카드가 아래서 올라오는 효과
document.addEventListener('DOMContentLoaded', function () {

    // 페이지 로드 시 이미 보이는 요소 처리
    document.querySelectorAll('.scroll-reveal').forEach(function(el) {
        var rect = el.getBoundingClientRect();
        if (rect.top < window.innerHeight) {
            el.classList.add('visible');
        }
    });



    // scroll-reveal 클래스 붙은 요소들 감지
    var observer = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                observer.unobserve(entry.target); // 한 번만 실행
            }
        });
    }, { threshold: 0.1 });

    document.querySelectorAll('.scroll-reveal').forEach(function (el) {
        observer.observe(el);
    });

});