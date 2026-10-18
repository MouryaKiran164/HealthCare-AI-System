
document.addEventListener("DOMContentLoaded", function () {

    const message = document.getElementById("message");

    function showMessage(text, color) {
        if (message) {
            message.textContent = text;
            message.style.color = color;
        }
    }

    // REGISTER
    const registerForm = document.getElementById("registerForm");

    if (registerForm) {
        registerForm.addEventListener("submit", function (event) {
            event.preventDefault();

            const name = document.getElementById("name").value.trim();
            const email = document.getElementById("email").value.trim().toLowerCase();
            const password = document.getElementById("password").value;
            const confirmPassword = document.getElementById("confirmPassword").value;

            if (password.length < 8) {
                showMessage("Password must contain at least 8 characters.", "red");
                return;
            }

            if (password !== confirmPassword) {
                showMessage("Passwords do not match.", "red");
                return;
            }

            try {
                localStorage.setItem("userName", name);
                localStorage.setItem("userEmail", email);
                localStorage.setItem("userPassword", password);
                localStorage.removeItem("loggedIn");

                showMessage("Registration successful! Redirecting to login...", "green");

                setTimeout(function () {
                    window.location.href = "index.html";
                }, 1000);

            } catch (error) {
                showMessage("Unable to save registration details.", "red");
            }
        });
    }

    // LOGIN
    const loginForm = document.getElementById("loginForm");

    if (loginForm) {
        loginForm.addEventListener("submit", function (event) {
            event.preventDefault();

            const email = document.getElementById("email").value.trim().toLowerCase();
            const password = document.getElementById("password").value;

            const savedEmail = localStorage.getItem("userEmail");
            const savedPassword = localStorage.getItem("userPassword");

            if (!savedEmail || savedPassword === null) {
                showMessage("Please register an account first.", "red");
                return;
            }

            if (email !== savedEmail || password !== savedPassword) {
                showMessage("Incorrect email or password.", "red");
                return;
            }

            localStorage.setItem("loggedIn", "true");

            showMessage("Login successful! Opening services...", "green");

            setTimeout(function () {
                window.location.href = "services.html";
            }, 700);
        });
    }

    // SHOW / HIDE PASSWORD
    const togglePassword = document.getElementById("togglePassword");
    const passwordInput = document.getElementById("password");

    if (togglePassword && passwordInput) {
        togglePassword.addEventListener("click", function () {
            const isPassword = passwordInput.type === "password";

            passwordInput.type = isPassword ? "text" : "password";
            togglePassword.textContent = isPassword ? "Hide" : "Show";
        });
    }

    // FORGOT PASSWORD
    const forgotLink = document.getElementById("forgotLink");

    if (forgotLink) {
        forgotLink.addEventListener("click", function (event) {
            event.preventDefault();
            showMessage("Password recovery is not configured yet.", "orange");
        });
    }

});
