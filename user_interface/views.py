from django.shortcuts import render, redirect
from .models import Users
from django.contrib import messages
from django.contrib.auth.hashers import make_password, check_password
import re
from django.core.mail import send_mail
from django.conf import settings
import random

# Create your views here.
def home(request):
    user_id = request.session.get('user_id')
    if user_id:
        return redirect('dashboard')
    return render(request, "home.html")

def signup_view(request):
    if request.method == "POST":
        first_name = request.POST.get('firstName')
        last_name = request.POST.get('lastName')
        email = request.POST.get('email')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirmPassword')
        terms = request.POST.get('terms') == 'on'
        newsletter = request.POST.get('newsletter') == 'on'

        # Password match check
        if password != confirm_password:
            messages.error(request, "Passwords do not match.")
            return render(request, "signup.html", {
                'firstName': first_name,
                'lastName': last_name,
                'email': email,
                'newsletter': newsletter,
                'terms': terms
            })

        # Terms check
        if not terms:
            messages.error(request, "You must agree to the terms.")
            return render(request, "signup.html", {
                'firstName': first_name,
                'lastName': last_name,
                'email': email,
                'newsletter': newsletter
            })

        # Password strength check
        password_pattern = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*#?&])[A-Za-z\d@$!%*#?&]{8,}$'
        if not re.match(password_pattern, password):
            messages.error(
                request,
                "Password must be at least 8 characters, include uppercase, lowercase, number, and special character."
            )
            return render(request, "signup.html", {
                'firstName': first_name,
                'lastName': last_name,
                'email': email,
                'newsletter': newsletter,
                'terms': terms
            })

        # Save user
        hashed_password = make_password(password)
        Users.objects.create(
            First_name=first_name,
            Last_name=last_name,
            Email=email,
            Password=hashed_password,
            subscribed_to_newsletter=newsletter,
            agreed_to_terms=terms
        )

        messages.success(request, "Account created successfully!")
        return redirect('login')

    return render(request, "signup.html")

def login_view(request):
    if request.method == "POST":
        email = request.POST.get('email')
        password = request.POST.get('password')

        try:
            user = Users.objects.get(Email=email)
            if check_password(password, user.Password):
                # Set session
                request.session['user_id'] = user.id
                request.session['user_name'] = user.First_name
                return redirect('dashboard')
            else:
                messages.error(request, "Incorrect password.")
        except Users.DoesNotExist:
            messages.error(request, "User with this email does not exist.")

        # Return form with the email field pre-filled
        return render(request, "login.html", {
            'email': email
        })

    return render(request, "login.html")

def logout_view(request):
    request.session.flush()
    messages.success(request, "Logged out successfully.")
    return redirect('home')

otp_storage = {}

def forgot_password_flow(request):
    step = request.POST.get("step", "email")  # Default step: email
    email = request.POST.get("email", "")
    
    if request.method == "POST":
        # STEP 1: User enters email
        if step == "email":
            try:
                user = Users.objects.get(Email=email)
                otp = str(random.randint(100000, 999999))
                otp_storage[email] = otp

                send_mail(
                    'Your Password Reset OTP',
                    f'Your OTP for password reset is: {otp}',
                    settings.DEFAULT_FROM_EMAIL,
                    [email],
                    fail_silently=False,
                )
                messages.success(request, "OTP sent to your email.")
                step = "otp"
            except Users.DoesNotExist:
                messages.error(request, "Email not found.")

        # STEP 2: Verify OTP
        elif step == "otp":
            otp = request.POST.get("otp", "")
            if otp_storage.get(email) == otp:
                request.session["reset_email"] = email
                step = "reset"
            else:
                messages.error(request, "Invalid OTP.")
                step = "otp"

        # STEP 3: Reset Password
        elif step == "reset":
            new_password = request.POST.get("password", "")
            confirm_password = request.POST.get("confirm_password", "")
            email = request.session.get("reset_email")

            if new_password != confirm_password:
                messages.error(request, "Passwords do not match.")
                step = "reset"
            else:
                # Strong password validation
                password_pattern = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*#?&])[A-Za-z\d@$!%*#?&]{8,}$'
                if not re.match(password_pattern, new_password):
                    messages.error(
                        request,
                        "Password must be at least 8 characters, include uppercase, lowercase, number, and special character."
                    )
                    step = "reset"  # stay on reset step
                else:
                    try:
                        user = Users.objects.get(Email=email)
                        # Hash and save the password
                        user.Password = make_password(new_password)
                        user.save()

                        # Remove OTP after success
                        otp_storage.pop(email, None)

                        messages.success(request, "Password reset successfully.")
                        return redirect("login")
                    except Users.DoesNotExist:
                        messages.error(request, "User account not found.")
                        step = "email"

    return render(request, "forgot_password_flow.html", {"step": step, "email": email})