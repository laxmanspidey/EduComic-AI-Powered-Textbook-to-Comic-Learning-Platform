"""
Forms for the AI Comic Textbook application.
"""
from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from .models import Textbook


class TextbookUploadForm(forms.ModelForm):
    class Meta:
        model = Textbook
        fields = ['title', 'subject', 'grade_level', 'pdf_file']
        widgets = {
            'title': forms.TextInput(attrs={
                'class': 'input',
                'placeholder': 'e.g., Class 11 Physics NCERT',
            }),
            'subject': forms.TextInput(attrs={
                'class': 'input',
                'placeholder': 'e.g., Physics, Chemistry, History',
            }),
            'grade_level': forms.TextInput(attrs={
                'class': 'input',
                'placeholder': 'e.g., Grade 11, University',
            }),
            'pdf_file': forms.FileInput(attrs={
                'class': 'input',
                'accept': '.pdf',
            }),
        }

    def clean_pdf_file(self):
        f = self.cleaned_data.get('pdf_file')
        if f:
            if not f.name.lower().endswith('.pdf'):
                raise forms.ValidationError('Only PDF files are allowed.')
            if f.size > 50 * 1024 * 1024:  # 50 MB
                raise forms.ValidationError('File size must be under 50 MB.')
        return f


class RegisterForm(UserCreationForm):
    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(attrs={'class': 'input', 'placeholder': 'Email'}),
    )
    first_name = forms.CharField(
        max_length=50,
        required=False,
        widget=forms.TextInput(attrs={'class': 'input', 'placeholder': 'First Name'}),
    )

    class Meta:
        model = User
        fields = ['username', 'email', 'first_name', 'password1', 'password2']
        widgets = {
            'username': forms.TextInput(attrs={'class': 'input', 'placeholder': 'Username'}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['password1'].widget.attrs.update({'class': 'input', 'placeholder': 'Password'})
        self.fields['password2'].widget.attrs.update({'class': 'input', 'placeholder': 'Confirm Password'})
