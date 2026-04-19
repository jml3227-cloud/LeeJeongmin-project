const csrf = document.querySelector('input[name="csrf_token"]').value;
var mychart = null;

document.getElementById('predictForm').addEventListener('submit', function(e) {
    e.preventDefault();

    const formData = new FormData(document.getElementById('predictForm'));

    fetch('/churn/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json().then(json => ({ status: response.status, json })))
    .then(({ status, json }) => {

        // 에러 메세지 초기화
        document.querySelectorAll('.ajax-error').forEach(el => el.remove());

        // validation 실패
        if (status === 400 && json.validation_errors) {
            const fieldNames = {
                'total_min': '일주일 총 이용시간',
                'ott_services': '구독 중인 OTT 서비스',
                'devices': '이용 기기',
                'content_types': '시청 콘텐츠 유형',
                'family_type': '가족구성',
                'monthly_fee_code': '월 구독료',
                'recommend_view': '추천 시청 정도',
                'search_view': '검색 시청 정도',
                'binge_watch': '몰아보기 정도'
            };
            const form = document.getElementById('predictForm');
            Object.entries(json.validation_errors).forEach(([field, errs]) => {
                const div = document.createElement('div');
                div.className = 'alert alert-danger ajax-error';
                div.textContent = (fieldNames[field] || field) + ': ' + errs.join(', ');
                form.prepend(div);
            });
            window.scrollTo({ top: 0, behavior: 'smooth' });
            return;
        }

        // 카드 테두리 전부 초기화
        document.getElementById('card-high').className = 'card h-100';
        document.getElementById('card-mid').className = 'card h-100';
        document.getElementById('card-low').className = 'card h-100';

        // 해당 그룹 배지 + 테두리 표시
        if (json.freq_group === 2) {
            document.getElementById('badge-high').style.display = 'block';
            document.getElementById('card-high').className = 'card h-100 border-primary';
        } else if (json.freq_group === 1) {
            document.getElementById('badge-mid').style.display = 'block';
            document.getElementById('card-mid').className = 'card h-100 border-warning';
        } else {
            document.getElementById('badge-low').style.display = 'block';
            document.getElementById('card-low').className = 'card h-100 border-danger';
        }

        // 인사이트 텍스트
        document.getElementById('insightText').textContent = json.insight;

        // 그룹 라벨
        document.getElementById('groupLabelText').textContent = json.group_label;

        // 원인 분석
        const reasonsList = document.getElementById('reasonsList');
        reasonsList.innerHTML = '';
        json.reasons.forEach(function(reason) {
            const li = document.createElement('li');
            li.textContent = reason;
            li.className = 'mb-2';
            reasonsList.appendChild(li);
        });

        // 결과 섹션 보이게 + 스크롤
        document.getElementById('resultSection').style.display = 'block';
        document.getElementById('resultSection').scrollIntoView({ behavior: 'smooth' });

        // 차트
        fetch('/churn/chart-data')
        .then(response => response.json())
        .then(chartJson => {
            var offset = [0, 0, 0];
            if (json.freq_group === 0) offset[0] = 40;
            if (json.freq_group === 1) offset[1] = 40;
            if (json.freq_group === 2) offset[2] = 40;

            if (mychart === null) {
                var ctx = document.getElementById('pieChart').getContext('2d');
                mychart = new Chart(ctx, {
                    type: 'pie',
                    data: {
                        labels: ['저빈도 (이탈위험)', '중빈도 (관찰)', '고빈도(안정)'],
                        datasets: [{
                            label: '그룹 분포',
                            data: [chartJson.low, chartJson.mid, chartJson.high],
                            backgroundColor: ['#dc3545', '#ffc107', '#007bff'],
                            offset: offset,
                            hoverOffset: 10
                        }]
                    },
                    options: {
                        plugins: {
                            legend: { display: false }
                        },
                        elements: {
                            arc: {
                                offset: offset
                            }
                        }
                    }
                });
            } else {
                mychart.data.datasets[0].offset = offset;
                mychart.update();
            }

            document.getElementById('legend-low').textContent = chartJson.low + '명';
            document.getElementById('legend-mid').textContent = chartJson.mid + '명';
            document.getElementById('legend-high').textContent = chartJson.high + '명';
        });
    });
});

const recommendSelect = document.getElementById('recommend_view');
const searchSelect = document.getElementById('search_view');

const allOptions = [
  { value: '1', text: '전혀 그렇지 않다' },
  { value: '2', text: '그렇지 않다' },
  { value: '3', text: '보통이다' },
  { value: '4', text: '그렇다' },
  { value: '5', text: '매우 그렇다' }
];

function updateOptions(changedSelect, otherSelect) {
  const selectedVal = parseInt(changedSelect.value);
  const currentOtherVal = otherSelect.value;

  otherSelect.innerHTML = '';

  allOptions.forEach(function(opt) {
    if (selectedVal >= 4 && (opt.value === '4' || opt.value === '5')) {
      return;
    }
    const option = document.createElement('option');
    option.value = opt.value;
    option.text = opt.text;
    otherSelect.appendChild(option);
  });

  if (otherSelect.querySelector('option[value="' + currentOtherVal + '"]')) {
    otherSelect.value = currentOtherVal;
  }
}

recommendSelect.addEventListener('change', function() {
  updateOptions(recommendSelect, searchSelect);
});

searchSelect.addEventListener('change', function() {
  updateOptions(searchSelect, recommendSelect);
});