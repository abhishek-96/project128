package com.example.abhis.project128;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.text.method.ScrollingMovementMethod;
import android.widget.Button;
import android.view.View;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import com.android.volley.DefaultRetryPolicy;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;

import java.util.HashMap;
import java.util.Map;

public class MainActivity extends AppCompatActivity {

    private Button SubmitButton;
    private TextView inputtextview,outputtextview;
    private ProgressBar loading_circle;

    private static final String URL = "http://192.168.120.132:8080/summary/";
    private String input;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        SubmitButton = (Button) findViewById(R.id.button);
        inputtextview = (TextView) findViewById(R.id.iptextView);
        outputtextview = (TextView) findViewById(R.id.optextView);
        loading_circle = (ProgressBar) findViewById(R.id.progressBar);



        SubmitButton.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View view) {

                loading_circle.setVisibility(View.VISIBLE);
                StringRequest stringRequest = new StringRequest(Request.Method.POST, URL,

                        new Response.Listener<String>() {
                            @Override
                            public void onResponse(String response) {

                                outputtextview.setMovementMethod(new ScrollingMovementMethod());
                                outputtextview.setText(response);
                                loading_circle.setVisibility(View.GONE);
                            }
                        },
                        new Response.ErrorListener() {
                            @Override
                            public void onErrorResponse(VolleyError error) {

                                Toast.makeText(MainActivity.this, error.toString(), Toast.LENGTH_LONG).show();
                                loading_circle.setVisibility(View.GONE);

                            }
                        }) {
                    @Override
                    protected Map<String, String> getParams() {

                        Map<String, String> params = new HashMap<String, String>();
                        input = inputtextview.getText().toString();
                        String temp = "<TEXT>\n",temp1;
                        temp1 = temp + input;
                        temp = "\n<TEXT>";
                        temp1 = temp1 + temp;
                        input = temp1;
                        params.put("message", input);
                        return params;
                    }
                };



                RequestQueue requestQueue = Volley.newRequestQueue(MainActivity.this);
                requestQueue.add(stringRequest);

            }
        });




    }
}
