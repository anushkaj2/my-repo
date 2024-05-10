document.addEventListener('DOMContentLoaded', function() {
    function fetchPathogens() {
        fetch('backend.php?read=1')
            .then(response => response.json())
            .then(data => {
                const pathogenList = document.getElementById('pathogenList');
                pathogenList.innerHTML = '';
                data.forEach(pathogen => {
                    const pathogenItem = document.createElement('div');
                    pathogenItem.className = 'pathogenItem';
                    pathogenItem.innerHTML = `
                        <p>Pathogen ID: ${pathogen.pathogen_id}</p>
                        <p>Ideal Temperature: ${pathogen.ideal_temperature}</p>
                        <p>Incubation Period: ${pathogen.incubation_period}</p>
                        <p>Pathogenicity: ${pathogen.pathogenicity}</p>
                        <p>Transmission Mode: ${pathogen.transmission_mode}</p>

                    `;
                    pathogenList.appendChild(pathogenItem);
                });
            });
    }

    document.getElementById('deleteForm').addEventListener('submit', function(event) {
        event.preventDefault();
        const formData = new FormData(this);
        const pathogen_id = formData.get('pathogen_id');

        deletePathogen(pathogen_id);
    });

    window.deletePathogen = function(pathogen_id) {
        if (confirm('Are you sure you want to delete this pathogen?')) {
            const formData = new FormData();
            formData.append('pathogen_id', pathogen_id);
            formData.append('delete', 1);
            fetch('backend.php', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.text())
                .then(data => {
                    alert(data);
                    fetchPathogens();
                });
        }
    }

    fetchPathogens();
});
