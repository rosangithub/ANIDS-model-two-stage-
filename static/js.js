// Optional JS functionality, like validations
document.querySelector('form').addEventListener('submit', function(e) {
    // Example: validate if all fields are filled
    const inputs = document.querySelectorAll('input');
    for (let input of inputs) {
        if (input.value.trim() === '') {
            alert('Please fill all fields!');
            e.preventDefault();
            return;
        }
    }
});
