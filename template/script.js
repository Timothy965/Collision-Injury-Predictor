document.addEventListener('DOMContentLoaded', function () {
    const predictButton = document.getElementById('predictButton');
    const predictedInjuryElement = document.getElementById('predictedInjury');

    document.addEventListener('DOMContentLoaded', function () {
        const predictButton = document.getElementById('predictButton');
        const predictedInjuryElement = document.getElementById('predictedInjury');

        predictButton.addEventListener('click', async function () {
            const year = document.getElementById('year').value;
            const street1 = document.getElementById('street1').value;
            const street2 = document.getElementById('street2').value;
            const roadClass = document.getElementById('roadClass').value;
            const district = document.getElementById('district').value;
            const locCoord = document.getElementById('locCoord').value;
            const traffCtl = document.getElementById('traffCtl').value;
            const visibility = document.getElementById('visibility').value;
            const light = document.getElementById('light').value;
            const rdsfCond = document.getElementById('rdsfCond').value;
            const impactType = document.getElementById('impactType').value;
            const invType = document.getElementById('invType').value;
            const invAge = document.getElementById('invAge').value;
            const vehType = document.getElementById('vehType').value;
            const manoeuver = document.getElementById('manoeuver').value;
            const drivAct = document.getElementById('drivAct').value;
            const drivCond = document.getElementById('drivCond').value;
            const pedType = document.getElementById('pedType').value;
            const pedAct = document.getElementById('pedAct').value;
            const pedCond = document.getElementById('pedCond').value;
            const cyclistType = document.getElementById('cyclistType').value;
            const cycAct = document.getElementById('cycAct').value;
            const cycCond = document.getElementById('cycCond').value;
            const pedestrian = document.getElementById('pedestrian').value;
            const cyclist = document.getElementById('cyclist').value;
            const automobile = document.getElementById('automobile').value;
            const motorcycle = document.getElementById('motorcycle').value;
            const truck = document.getElementById('truck').value;
            const trsnCityVeh = document.getElementById('trsnCityVeh').value;
            const emergVeh = document.getElementById('emergVeh').value;
            const passenger = document.getElementById('passenger').value;
            const speeding = document.getElementById('speeding').value;
            const agDriv = document.getElementById('agDriv').value;
            const redLight = document.getElementById('redLight').value;
            const alcohol = document.getElementById('alcohol').value;
            const disability = document.getElementById('disability').value;
            const neighbourhood158 = document.getElementById('neighbourhood158').value;
            const month = document.getElementById('month').value;
            const dayOfWeek = document.getElementById('dayOfWeek').value;
            const isRushHr = document.getElementById('isRushHr').value;

            const data = {
                'YEAR': [year],
                'STREET1': [street1],
                'STREET2': [street2],
                'ROAD_CLASS': [roadClass],
                'DISTRICT': [district],
                'LOCCOORD': [locCoord],
                'TRAFFCTL': [traffCtl],
                'VISIBILITY': [visibility],
                'LIGHT': [light],
                'RDSFCOND': [rdsfCond],
                'IMPACTYPE': [impactType],
                'INVTYPE': [invType],
                'INVAGE': [invAge],
                'VEHTYPE': [vehType],
                'MANOEUVER': [manoeuver],
                'DRIVACT': [drivAct],
                'DRIVCOND': [drivCond],
                'PEDTYPE': [pedType],
                'PEDACT': [pedAct],
                'PEDCOND': [pedCond],
                'CYCLISTYPE': [cyclistType],
                'CYCACT': [cycAct],
                'CYCCOND': [cycCond],
                'PEDESTRIAN': [pedestrian],
                'CYCLIST': [cyclist],
                'AUTOMOBILE': [automobile],
                'MOTORCYCLE': [motorcycle],
                'TRUCK': [truck],
                'TRSN_CITY_VEH': [trsnCityVeh],
                'EMERG_VEH': [emergVeh],
                'PASSENGER': [passenger],
                'SPEEDING': [speeding],
                'AG_DRIV': [agDriv],
                'REDLIGHT': [redLight],
                'ALCOHOL': [alcohol],
                'DISABILITY': [disability],
                'NEIGHBOURHOOD_158': [neighbourhood158],
                'MONTH': [month],
                'DAY_OF_WEEK': [dayOfWeek],
                'IS_RUSH_HR': [isRushHr]
            };

            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data),
                });

                const result = await response.json();
                predictedInjuryElement.textContent = result.predictedInjury;
                document.getElementById('predictionResult').style.display = 'block';
            } catch (error) {
                console.error('Error predicting injury:', error);
            }

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data),
            });

            const result = await response.json();
            predictedInjuryElement.textContent = result.predictedInjury;
            document.getElementById('predictionResult').style.display = 'block';
        } catch (error) {
            console.error('Error predicting injury:', error);
        }
    });
});

